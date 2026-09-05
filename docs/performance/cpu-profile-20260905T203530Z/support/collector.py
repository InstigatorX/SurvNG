#!/usr/bin/env python3
"""Read-only SurvNG baseline; stdlib only, no application imports or inference."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent
HZ = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')
STOP = False

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def run(*args):
    return subprocess.check_output(args, text=True, timeout=10).strip()

def clean(value):
    if isinstance(value, dict):
        return {k: ('[redacted]' if re.search(r'password|token|secret|credential|stream_url|api_key', k, re.I) else clean(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'\b(?:https?|rtsps?|rtmp)://[^\s\"<>]+', '[redacted-url]', value)
    return value

def write(path, value):
    with path.open('a') as out:
        out.write(json.dumps(clean(value), separators=(',', ':')) + '\n')

def status(sock):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(3)
        client.connect(sock)
        client.sendall(b'{"version":1,"command":"status"}\n')
        with client.makefile('rb') as stream:
            raw = stream.readline(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024 or not raw.endswith(b'\n'):
        raise ValueError('invalid snapshot size')
    result = json.loads(raw)
    if not result.get('ok') or not isinstance(result.get('status'), dict):
        raise ValueError('snapshot unavailable')
    return result['status']

def proc(pid):
    base = Path('/proc') / str(pid)
    raw = (base / 'stat').read_text()
    end = raw.rindex(')')
    fields = raw[end + 2:].split()
    io = {}
    try:
        io = {k: int(v) for k, v in (line.split(':') for line in (base / 'io').read_text().splitlines())}
    except (OSError, ValueError):
        pass
    return dict(pid=pid, ppid=int(fields[1]), comm=raw[raw.index('(')+1:end],
                start_ticks=int(fields[19]), cpu_seconds=(int(fields[11])+int(fields[12]))/HZ,
                rss_bytes=int(fields[21])*PAGE, threads=int(fields[17]), io=io)

def identity():
    values = run('systemctl', 'show', 'survng.service', '-p', 'MainPID', '-p', 'ControlGroup', '-p', 'ActiveEnterTimestamp', '-p', 'FragmentPath', '-p', 'WorkingDirectory')
    result = dict(line.split('=', 1) for line in values.splitlines())
    pid = int(result['MainPID'])
    base = Path('/proc') / str(pid)
    result['exe'] = os.readlink(base/'exe')
    result['cwd'] = os.readlink(base/'cwd')
    result['process'] = proc(pid)
    # Only read specific non-secret environment values; never save the environment.
    env = dict(item.split(b'=', 1) for item in (base/'environ').read_bytes().split(b'\0') if b'=' in item)
    config = Path(os.fsdecode(env.get(b'SURVNG_CONFIG_PATH', b'config.json')))
    if not config.is_absolute():
        config = Path(result['cwd'])/config
    result['config_path'] = str(config)
    result['socket'] = os.fsdecode(env.get(b'SURVNG_OBSERVABILITY_SOCKET', b'/run/survng/observability.sock'))
    result['checkout_commit'] = run('git', '-C', result['cwd'], 'rev-parse', 'HEAD')
    result['checkout_status'] = run('git', '-C', result['cwd'], 'status', '--short')
    result['deployment_commit_assessment'] = 'checkout hash; inferred loaded version from clean tree and pre-start reflog; no runtime build identifier'
    result['reflog'] = run('git', '-C', result['cwd'], 'reflog', '-5', '--date=iso')
    raw = config.read_bytes()
    result['persisted_config_sha256'] = hashlib.sha256(raw).hexdigest()
    result['config_mtime_utc'] = dt.datetime.fromtimestamp(config.stat().st_mtime, dt.timezone.utc).isoformat()
    cfg = json.loads(raw)
    numeric = lambda d: {k:v for k,v in d.items() if isinstance(v, (int,float,bool)) or v is None}
    result['persisted_settings_not_runtime_verified'] = {
        'motion_qualification': {**numeric(cfg.get('motion_qualification', {})), **{k:v for k,v in cfg.get('motion_qualification', {}).items() if k in ('mode','sensitivity','stationary_object_tolerance')}},
        'detector': {**numeric(cfg.get('detector', {})), **{k:Path(str(v)).name for k,v in cfg.get('detector', {}).items() if k in ('model_path','model_xml','backend','device')}},
        'cameras': [{k:v for k,v in c.items() if k in ('id','enabled','record','record_sub','motion_qualification')} for c in cfg.get('cameras', [])],
    }
    return result

def sample_processes(group, previous, now):
    pids = set()
    for path in group.rglob('cgroup.procs'):
        try:
            pids.update(map(int, path.read_text().split()))
        except OSError:
            pass
    rows = []
    for pid in sorted(pids | {os.getpid()}):
        try:
            row = proc(pid)
            key = (pid, row['start_ticks'])
            old = previous.get(key)
            row['scope'] = 'collector' if pid == os.getpid() else 'service_cgroup'
            if old:
                elapsed = now-old[0]
                row['interval_seconds'] = elapsed
                row['cpu_percent_one_core'] = 100*(row['cpu_seconds']-old[1]['cpu_seconds'])/elapsed
                row['io_delta'] = {k: v-old[1]['io'][k] for k,v in row['io'].items() if k in old[1]['io'] and v >= old[1]['io'][k]}
            previous[key] = (now, row)
            rows.append(row)
        except (OSError, ValueError, ProcessLookupError):
            continue
    active = {(r['pid'],r['start_ticks']) for r in rows}
    for key in list(previous):
        if key not in active:
            del previous[key]
    cg = {}
    for name in ('cpu.stat','memory.current','memory.events','io.stat','cpu.pressure','memory.pressure','io.pressure'):
        try:
            cg[name] = (group/name).read_text().strip()
        except OSError:
            cg[name] = None
    return {'processes':rows,'cgroup':cg,'loadavg':Path('/proc/loadavg').read_text().strip(),
            'host_cpu':Path('/proc/stat').read_text().splitlines()[0],
            'meminfo':{k:v.strip() for k,v in (x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','MemTotal','SwapFree','SwapTotal','Dirty','Writeback')}}

def collect(args):
    global STOP
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'collector.pid').write_text(str(os.getpid())+'\n')
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__('STOP', True))
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__('STOP', True))
    meta = identity()
    meta.update(started_at=utc(), duration_seconds=args.duration, process_interval_seconds=5,
                application_interval_seconds=15, cpu_count=os.cpu_count(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(args.output/'metadata.jsonl', meta)
    group = Path('/sys/fs/cgroup')/meta['ControlGroup'].lstrip('/')
    previous = {}
    prior_app = None
    start = time.monotonic()
    next_proc = next_app = start
    app_interval = 15
    while not STOP and time.monotonic()-start < args.duration:
        now = time.monotonic()
        if now >= next_proc:
            began = time.monotonic()
            data = sample_processes(group, previous, now)
            data.update(timestamp=utc(), monotonic=now, collection_ms=(time.monotonic()-began)*1000)
            write(args.output/'resources.jsonl', data)
            next_proc = now+5
        if now >= next_app:
            began = time.monotonic()
            try:
                snap = status(meta['socket'])
                # Logs are already server-redacted; additionally remove any URL text.
                counters = {'detector.'+k:v for k,v in snap.get('detector',{}).get('runtime',{}).items() if k in ('total_inferences','failed_inferences')}
                for c in snap.get('cameras',[]):
                    counters.update({c['id']+'.'+k:v for k,v in c.get('tracking',{}).items() if k in ('capacity_requests','capacity_waits','capacity_timeouts','coverage_gap_count')})
                instance = snap.get('process',{}).get('instance_id')
                workers = sorted((r['pid'],r['start_ticks']) for r in sample_processes(group, {}, now)['processes'] if r['scope']=='service_cgroup')
                epoch = (instance, workers)
                delta = {}
                if prior_app and epoch == prior_app[0]:
                    delta = {k:(v-prior_app[2][k] if v >= prior_app[2][k] else None) for k,v in counters.items() if k in prior_app[2]}
                data = {'timestamp':utc(),'monotonic':now,'status':snap,'counter_deltas':delta,
                        'delta_interval_seconds':now-prior_app[1] if prior_app and epoch==prior_app[0] else None,
                        'reset_or_process_tree_change':bool(prior_app and epoch!=prior_app[0])}
                prior_app = (epoch,now,counters)
            except Exception as error:
                data = {'timestamp':utc(),'monotonic':now,'error_type':type(error).__name__}
                prior_app = None
            data['request_ms'] = (time.monotonic()-began)*1000
            if data['request_ms'] > 250 or 'error_type' in data:
                app_interval = min(120,app_interval*2)
            data['next_interval_seconds'] = app_interval
            write(args.output/'application.jsonl', data)
            next_app = time.monotonic()+app_interval
        time.sleep(min(0.5,max(0.01,min(next_proc,next_app)-time.monotonic())))
    write(args.output/'completion.jsonl', {'ended_at':utc(),'elapsed_seconds':time.monotonic()-start,'stopped_early':STOP,'self_user_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime,'self_system_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_stime})

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('collect'); p.add_argument('--output',type=Path,required=True); p.add_argument('--duration',type=float,default=1800)
    p = sub.add_parser('mark'); p.add_argument('--output',type=Path,required=True); p.add_argument('--camera',required=True); p.add_argument('--scenario',required=True); p.add_argument('--subject',required=True); p.add_argument('--phase',choices=('start','end','note'),required=True); p.add_argument('--at'); p.add_argument('--note',default='')
    args = parser.parse_args()
    if args.command=='collect':
        collect(args)
    else:
        write(args.output/'observations.jsonl', dict(timestamp=utc(),observed_at=args.at or utc(),camera_id=args.camera,scenario=args.scenario,expected_subject=args.subject,phase=args.phase,note=args.note))

if __name__=='__main__':
    main()
