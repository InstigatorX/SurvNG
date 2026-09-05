#!/usr/bin/env python3
"""Aligned process and thread CPU from independent one-second /proc samples."""
from collections import defaultdict
import datetime as dt
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
def rows(p):return [json.loads(line) for line in p.read_text().splitlines()]
def main():
    data=rows(ROOT/'cpu.jsonl'); meta=rows(ROOT/'metadata.jsonl')[0]
    w=json.loads((ROOT/'profile-window.json').read_text())
    if w['status']=='running':raise SystemExit('Profile still running')
    lo=w['started_monotonic']; hi=w.get('ended_monotonic')
    if hi is None:
        end=dt.datetime.fromisoformat(w['ended_at']).timestamp()
        hi=data[-1]['monotonic']+end-dt.datetime.fromisoformat(data[-1]['timestamp']).timestamp()
    elapsed=total=0; n=0; pid_cpu=defaultdict(float); tid_cpu=defaultdict(float); categories=defaultdict(float)
    boundaries=0; intervals=[]
    for a,b in zip(data,data[1:]):
        if a['monotonic']<lo or b['monotonic']>hi:continue
        seconds=b['monotonic']-a['monotonic']
        cg=lambda x:int(dict(line.split() for line in x['cgroup']['cpu.stat'].splitlines())['usage_usec'])/1e6
        dcg=cg(b)-cg(a)
        old={(p['pid'],p['start_ticks']):p for p in a['processes'] if p['scope']=='service_cgroup'}
        new={(p['pid'],p['start_ticks']):p for p in b['processes'] if p['scope']=='service_cgroup'}
        ma=[key for key in old if key[0]==int(meta['MainPID'])]; mb=[key for key in new if key[0]==int(meta['MainPID'])]
        if dcg<0 or seconds<=0 or not ma or ma!=mb:boundaries+=1;continue
        elapsed+=seconds;total+=dcg;n+=1;intervals.append(seconds)
        for key,p in new.items():
            prior=old.get(key)
            delta=p['cpu_seconds']-prior['cpu_seconds'] if prior else p['cpu_seconds'] if p['start_ticks']/meta['clock_ticks']>=a['monotonic'] else 0
            if delta<0:continue
            pid_cpu[(p['pid'],p['start_ticks'],p['comm'])]+=delta
            categories['main' if p['pid']==int(meta['MainPID']) else p['comm']]+=delta
        before={(t['tid'],t['start_ticks']):t for t in a['main_threads']}
        for t in b['main_threads']:
            p=before.get((t['tid'],t['start_ticks']))
            delta=t['cpu_seconds']-p['cpu_seconds'] if p else t['cpu_seconds'] if t['start_ticks']/meta['clock_ticks']>=a['monotonic'] else 0
            if delta>=0:tid_cpu[(t['tid'],t['start_ticks'],t['comm'])]+=delta
    threads=[{'tid':k[0],'start_ticks':k[1],'comm':k[2],'cpu_seconds':v,'cpu_cores':v/elapsed,'fraction_of_main_cpu':v/categories['main'] if categories['main'] else None} for k,v in sorted(tid_cpu.items(),key=lambda item:-item[1])]
    output={'profile_window':w,'resource_interval_count':n,'covered_seconds':elapsed,'profile_wrapper_seconds':hi-lo,'excluded_boundary_intervals':boundaries,'service_cpu_seconds':total,'service_cpu_cores':total/elapsed,'category_cpu_seconds':dict(categories),'category_cpu_cores':{k:v/elapsed for k,v in categories.items()},'unattributed_service_cpu_seconds':total-sum(pid_cpu.values()),'main_thread_cpu_seconds_sum':sum(tid_cpu.values()),'main_process_minus_observed_threads_cpu_seconds':categories['main']-sum(tid_cpu.values()),'threads':threads,'processes':[{'pid':k[0],'start_ticks':k[1],'comm':k[2],'cpu_seconds':v,'cpu_cores':v/elapsed} for k,v in sorted(pid_cpu.items(),key=lambda item:-item[1])],'notes':['Only whole one-second resource intervals inside the profiler wrapper are included; actual sample coverage may differ from nominal60s.','Cgroup total includes service processes; main-process CPU includes its threads. These are nested scopes, not additive totals.','Exited-between-poll processes/threads remain unattributed; independently read counters have quantization/skew.','Thread comm does not establish Python/OpenCV ownership; correlate native TIDs with profiler records where possible.']}
    (ROOT/'cpu-summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('covered_seconds','service_cpu_cores','category_cpu_cores','main_thread_cpu_seconds_sum','main_process_minus_observed_threads_cpu_seconds')}))
if __name__=='__main__':main()
