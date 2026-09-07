"""Bounded operator API session; token kept only in process memory, never logged.
Run interactively with a private terminal, enter token, then JSON request lines.
Only allowlisted GETs and history-preserving Compare POSTs are supported.
"""
import os, sys, json, time, termios, sqlite3, subprocess, hashlib
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8088/survng'
os.umask(0o077)
def save(name, data):
    path = ROOT / 'private' / name
    if path.exists(): raise RuntimeError('refusing to overwrite evidence')
    path.write_text(json.dumps(data, indent=2) + '\n')
def history():
    with sqlite3.connect('file:DEPLOYMENT/runtime/database/survng.sqlite3?mode=ro', uri=True, timeout=2) as db:
        db.execute('PRAGMA query_only=ON')
        deadline=time.monotonic()+4
        db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
        return [{'id':r[0], 'event_id':r[1], 'camera_id':r[2], 'sha256':hashlib.sha256(r[3].encode()).hexdigest()} for r in db.execute('select id,event_id,camera_id,json_array(result_json,verdict,reviewed_at,created_at) from tracking_comparisons order by id')]
def health():
    status = json.loads(subprocess.check_output(['DEPLOYMENT/survngctl','status','--compact','--socket','/run/survng/observability.sock'],text=True,timeout=15))
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    status['experiment_os']={'memory_available':mem['MemAvailable'],'memory_total':mem['MemTotal'], 'cpu_pressure':Path('/proc/pressure/cpu').read_text(), 'memory_pressure':Path('/proc/pressure/memory').read_text()}
    return status
def main():
    if sys.stdin.isatty():
        attrs=termios.tcgetattr(sys.stdin);attrs[3] &= ~termios.ECHO;termios.tcsetattr(sys.stdin,termios.TCSANOW,attrs)
    print('TOKEN_INPUT_READY', flush=True)
    token=sys.stdin.readline().strip()
    print('SESSION_READY (credential withheld)', flush=True)
    initial=health();save('runtime-resume-before.json',initial);save('history-resume-before.json',history())
    captures=0; uncertain=False
    for line in sys.stdin:
        if line.strip()=='quit':break
        try:
            args=json.loads(line); path=args['path']; method=args.get('method','GET');name=args['name']
            if '/' in name or name.startswith('.'):raise ValueError('invalid evidence name')
            if method=='GET':
                if not any(path.startswith(prefix) for prefix in ('/api/incidents/feed?', '/api/incidents/search?', '/api/incidents/detail?', '/api/tracking-comparisons?')):raise ValueError('GET not allowlisted')
            elif method=='POST':
                import re
                match=re.fullmatch(r'/api/events/(\d+)/tracking-comparison\?duration_seconds=30&sampling_profile=fixed_2fps',path)
                if not match or captures>=8 or uncertain:raise ValueError('capture blocked by task limits or uncertain previous completion')
                corpus=json.loads((ROOT/'corpus.json').read_text()); event_id=int(match[1])
                case=next(c for c in corpus['selected'] if c['event_id']==event_id)
                previous=history();save(name+'.history-before.json',previous)
                if any(r['event_id']==event_id for r in previous):raise ValueError('existing comparison must not be overwritten')
                if sum(r['camera_id']==case['camera_id'] for r in previous)>=100:raise ValueError('history has no headroom')
                current=health();save(name+'.health-before.json',current)
                d=current['detector']; m=current['experiment_os'];base_d=initial['detector']
                if m['memory_available'] < max(2**30,m['memory_total']*.1):raise ValueError('memory floor')
                if not d['ready'] or d['runtime']['queue_depth'] or d['workers']['pending_requests'] or d['recorded_decode']['waiting']:raise ValueError('inference/decode backlog')
                if d['runtime']['failed_inferences']>base_d['runtime']['failed_inferences']:raise ValueError('new detector failure')
                if current['storage']['emergency'] or any(c['recording_enabled'] and not c['recording'] for c in current['cameras']):raise ValueError('storage/recording unhealthy')
                captures+=1
            else:raise ValueError('method not allowed')
            started=time.time();request=Request(BASE+path,headers={'Authorization':'Bearer '+token},method=method)
            try:
                with urlopen(request,timeout=150 if method=='POST' else 20) as response:
                    data=response.read(40*1024*1024+1)
                    if len(data)>40*1024*1024:raise ValueError('response limit')
                    payload=json.loads(data)
                    save(name+'.json',payload)
                    info={'status':response.status,'name':name,'elapsed_seconds':round(time.time()-started,3),'bytes':len(data),'keys':list(payload) if isinstance(payload,dict) else None}
                    if method=='POST':
                        info.update(comparison_id=payload.get('comparison_id'),replay_present=isinstance(payload.get('replay'),dict))
                        if isinstance(payload.get('replay'),dict):save(name+'.replay.json',payload['replay'])
                        after=history();save(name+'.history-after.json',after);save(name+'.health-after.json',health())
                        info['old_history_preserved']=all(row in after for row in previous)
                        if not info['old_history_preserved']:uncertain=True
                    save(name+'.request.json',info); print(json.dumps(info),flush=True)
            except HTTPError as error:
                info={'status':error.code,'name':name,'retry_after':error.headers.get('Retry-After')};save(name+'.request.json',info);print(json.dumps(info),flush=True)
            except (TimeoutError,OSError) as error:
                if method=='POST':uncertain=True
                info={'name':name,'error':type(error).__name__,'capture_completion_uncertain':uncertain};save(name+'.request.json',info);print(json.dumps(info),flush=True)
        except Exception as error:
            print(json.dumps({'error':type(error).__name__,'detail':str(error) if isinstance(error,(ValueError,RuntimeError,StopIteration)) else 'see operation context; no raw errors exposed'}),flush=True)
    token=None
    save('runtime-resume-after.json',health());save('history-resume-after.json',history())
    print('SESSION_CLOSED',flush=True)
if __name__=='__main__':main()
