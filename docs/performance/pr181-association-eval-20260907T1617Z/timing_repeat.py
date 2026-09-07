"""Single prespecified concerning timing case: one warmup, three alternating pairs.
Saved detections/embeddings only; no detector or encoder is instantiated.
"""
import json,sys,os,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'checkout'))
from survng.app.tracking_comparison import TrackingComparisonRunner
from survng.app.object_track.registry import build_builtin_object_tracker_registry
from survng.app.object_track.multicue import IMPLEMENTATION,HybridMultiCueObjectTracker
from survng.app.tracking_evaluation import validate_replay
os.umask(0o077)
replay=json.loads((ROOT/'private/capture-63156.replay.json').read_text());validate_replay(replay)
registry=build_builtin_object_tracker_registry();registry.register(IMPLEMENTATION,HybridMultiCueObjectTracker)
B='survng_hybrid';C=IMPLEMENTATION
runs=[]
for i,order in enumerate(((B,C),(C,B),(B,C),(C,B))):
    result=TrackingComparisonRunner.replay(replay,sampling_profile='fixed_2fps',tracker_registry=registry,implementations=order)
    assert all(not e.get('error') for e in result['engines'].values())
    assert result['engines'][B]['frame_observations']==result['engines'][C]['frame_observations']
    (ROOT/'results'/f'timing-63156-{i}.json').write_text(json.dumps(result,indent=2)+'\n')
    runs.append({'round':i,'warmup':i==0,'order':order,'baseline_ms_per_frame':result['engines'][B]['average_ms_per_frame'],'candidate_ms_per_frame':result['engines'][C]['average_ms_per_frame'],
                 'baseline_processing_ms':result['engines'][B]['processing_ms'],'candidate_processing_ms':result['engines'][C]['processing_ms']})
measured=runs[1:];bm=[x['baseline_ms_per_frame'] for x in measured];cm=[x['candidate_ms_per_frame'] for x in measured]
summary={'event_id':63156,'profile':'fixed_2fps','reason':'Largest initial positive absolute overhead +0.060ms/frame (+32%); bounded repeat checks indicative timing rather than declaring a regression from one sample.',
         'runs':runs,'baseline_median_ms':statistics.median(bm),'candidate_median_ms':statistics.median(cm),'baseline_range_ms':[min(bm),max(bm)],'candidate_range_ms':[min(cm),max(cm)],
         'paired_deltas_ms':[round(c-b,6) for b,c in zip(bm,cm)],'note':'Tracker update/deepcopy wall time, not process CPU or fleet/iGPU utilization. Rounded at source to 0.001ms/frame; tiny absolute differences and 3 rounds limit inference.'}
(ROOT/'timing-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
