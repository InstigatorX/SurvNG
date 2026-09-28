"""Physical source frames for integration tests that require real activity."""
import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from survng.app.scene_activity_evidence import scene_sample_records


def record_person_motion(store,event_id):
    event=store.get(event_id)
    objects=json.loads(event['objects_json'])
    raw=next((o for o in objects if o.get('label')=='person'),None)
    if raw is None:
        raw=next(o for batch in objects if batch.get('status')=='scene_observations' for o in batch['observations'] if o.get('label')=='person')
    at=datetime.fromisoformat(event['created_at']).timestamp()
    box=raw['box']; width=int(raw.get('detection_frame_width') or max(160,box['x2']+100)); height=int(raw.get('detection_frame_height') or max(100,box['y2']+50))
    source=raw.get('frame_source','legacy')
    observed=[]; samples=[]
    for i in range(2):
        b={k:float(v)+(10*i if k.startswith('x') else 0) for k,v in box.items()}
        item={**raw,'box':b,'captured_at_epoch':at+i,'frame_source':source,
              'detection_frame_width':width,'detection_frame_height':height,
              'scene_track_key':raw.get('scene_track_key','physical-proof')}
        observed.append(item)
        frame=np.zeros((height,width,3),dtype=np.uint8)
        frame[int(b['y1']):int(b['y2']),int(b['x1']):int(b['x2'])]=220
        samples.append(SimpleNamespace(offset=at+i,frame=frame,objects=[item],recording_path='',exact_timestamp=True))
    records=scene_sample_records(samples,observed,0,event['camera_id'],source=source)
    store.update_object_tracking(event_id,{'scene_observations':observed,'scene_samples':records,'state':'complete'})
