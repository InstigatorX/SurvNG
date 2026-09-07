"""Run only the locked offline Hybrid pair on immutable normal Compare captures."""
import os,json,sys,subprocess,hashlib,time,statistics
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
CHECKOUT=ROOT.parent/'checkout'
sys.path.insert(0,str(CHECKOUT))
from survng.app.tracking_evaluation import validate_replay,selected_frames
from survng.app.object_track import hybrid,multicue
from survng.app import tracking_evaluation
assert all(str(CHECKOUT) in m.__file__ for m in (hybrid,multicue,tracking_evaluation))
os.umask(0o077)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(CHECKOUT),'XDG_CACHE_HOME':str(ROOT/'private'),'TMPDIR':str(ROOT/'private')}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def main():
    corpus=json.loads((ROOT/'corpus.json').read_text());assert sha(ROOT/'corpus.json')==(ROOT/'private/corpus-lock-sha256.txt').read_text().strip()
    evidence=[];executions=[]
    for case in corpus['selected']:
        path=ROOT/'private'/f'capture-{case["event_id"]}.replay.json'
        if not path.exists():
            evidence.append({'event_id':case['event_id'],'status':'NO_CAPTURE'});continue
        digest=sha(path);replay=json.loads(path.read_text());validate_replay(replay)
        capture=json.loads((ROOT/'private'/f'capture-{case["event_id"]}.json').read_text())
        frames=replay['frames'];ts=[f['captured_at'] for f in frames];dt=[b-a for a,b in zip(ts,ts[1:])]
        ev={'event_id':case['event_id'],'replay_id':replay['replay_id'],'file_sha256':digest,'frame_count':len(frames),'detection_count':sum(len(f['detections']) for f in frames),
            'dimensions':[frames[0]['width'],frames[0]['height']],'first_epoch':ts[0],'last_epoch':ts[-1],'actual_cadence_seconds':{'min':min(dt),'median':statistics.median(dt),'max':max(dt)},
            'timestamp_source':replay['timestamp_source'],'source_pts_frames':replay['source_pts_frames'],'embeddings':sum('_tracking_embedding' in d for f in frames for d in f['detections']),
            'detector_provenance':replay['detector_identity'],'appearance_source':replay['appearance_source'],'appearance_failures':capture['appearance_failures'],
            'optional_capture_engine_errors':{k:v.get('error') for k,v in capture['engines'].items() if v.get('error')},'comparison_id':capture['comparison_id'],'profiles':{}}
        for profile in case['profiles']:
            selected=selected_frames(replay,profile)
            ev['profiles'][profile]={'frame_indexes':[f['frame_index'] for f in selected],'timestamps':[f['captured_at'] for f in selected],'count':len(selected),
                'supplied_embeddings':sum('_tracking_embedding' in d for f in selected for d in f['detections'])}
            name=f'{case["event_id"]}-{profile}';dest=ROOT/'results'/(name+'.json')
            assert not dest.exists(),'refusing to overwrite result'
            cmd=[sys.executable,'-m','survng.app.tracking_evaluation',str(path),'--association-cues','--profile',profile,'--output',str(dest)]
            started=time.monotonic()
            with (ROOT/'logs'/(name+'.log')).open('w') as log:
                try:run=subprocess.run(cmd,cwd=CHECKOUT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90);rc=run.returncode
                except subprocess.TimeoutExpired:rc=124
            execution={'event_id':case['event_id'],'profile':profile,'returncode':rc,'elapsed_seconds':time.monotonic()-started}
            if dest.exists():
                result=json.loads(dest.read_text());execution['engine_errors']={k:v.get('error') for k,v in result.get('engines',{}).items()}
                assert set(result['engines'])=={'survng_hybrid','survng_hybrid_multicue'}
                assert result['replay_id']==replay['replay_id']
                for engine in result['engines'].values():
                    if not engine.get('error'):assert [f['frame_index'] for f in engine['frame_observations']]==[f['frame_index'] for f in selected]
                execution['result_sha256']=sha(dest)
            executions.append(execution);print(json.dumps(execution),flush=True)
            assert sha(path)==digest,'raw replay modified'
        evidence.append(ev)
        save(ROOT/'private/replay-validation.json',evidence);save(ROOT/'executions.json',executions)
    print('COMPLETE',len(executions),flush=True)
if __name__=='__main__':main()
