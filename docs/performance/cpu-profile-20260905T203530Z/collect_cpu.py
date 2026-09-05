#!/usr/bin/env python3
"""Independent /proc process/thread accounting for the bounded profile windows."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'pre-deployment'))
from collector import identity, sample_processes, status, utc, write
STOP=False

def threads(pid):
    rows=[]
    for path in (Path('/proc')/str(pid)/'task').iterdir():
        try:
            raw=(path/'stat').read_text(); at=raw.rindex(')'); fields=raw[at+2:].split()
            rows.append({'tid':int(path.name),'comm':raw[raw.index('(')+1:at],
                         'state':fields[0],'start_ticks':int(fields[19]),
                         'cpu_seconds':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')})
        except (OSError,ValueError):continue
    return rows

def main():
    os.umask(0o077)
    (ROOT/'collector.pid').write_text(str(os.getpid())+'\n')
    signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True))
    meta=identity(); meta.update(started_at=utc(),duration_seconds=480,interval_seconds=1,clock_ticks=os.sysconf('SC_CLK_TCK'))
    write(ROOT/'metadata.jsonl',meta)
    try:write(ROOT/'status-before.jsonl',{'timestamp':utc(),'status':status(meta['socket'])})
    except Exception as error:write(ROOT/'status-before.jsonl',{'timestamp':utc(),'error_type':type(error).__name__})
    main_pid=int(meta['MainPID']); group=Path('/sys/fs/cgroup')/meta['ControlGroup'].lstrip('/')
    previous={}; start=time.monotonic(); due=start
    while not STOP and time.monotonic()-start<480:
        now=time.monotonic()
        if now<due:time.sleep(min(.1,due-now));continue
        began=time.monotonic(); data=sample_processes(group,previous,now)
        try:data['main_threads']=threads(main_pid)
        except OSError:data['main_threads']=[];data['main_thread_error']='original process unavailable'
        data.update(timestamp=utc(),monotonic=now,main_pid=main_pid,collection_ms=(time.monotonic()-began)*1000)
        write(ROOT/'cpu.jsonl',data);due=now+1
    checks={'timestamp':utc(),'collector_self_cpu_seconds':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'elapsed_seconds':time.monotonic()-start,'stop_requested':STOP}
    try:
        after=identity()
        checks['service_identity_config_checkout_unchanged']=all(after.get(k)==meta.get(k) for k in ('MainPID','ActiveEnterTimestamp','exe','cwd','checkout_commit','checkout_status','persisted_config_sha256'))
        checks['after_identity']=after
        write(ROOT/'status-after.jsonl',{'timestamp':utc(),'status':status(meta['socket'])})
    except Exception as error:checks['verification_error_type']=type(error).__name__
    write(ROOT/'completion.jsonl',checks)
if __name__=='__main__':main()
