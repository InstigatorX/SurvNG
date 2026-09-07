"""Freeze selection before any Compare capture or paired result is inspected."""
import json,sqlite3,time,os,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
os.umask(0o077)
media=json.loads((ROOT/'private/media-candidates.json').read_text())
specs=[('crossing_1',63156,'Two people simultaneously leaving the porch toward a shared narrow image edge; close-interaction candidate, not a proven crossing.','moderate'),
       ('crossing_2',63346,'Two people approaching the front porch together at night; simultaneously visible and approaching a shared entrance.','moderate'),
       ('occlusion_return',64897,'Two people move around a parked car; later frames show partial obstruction at the car/door. Suspected occlusion, return identity NOT established. Earlier actual child anchors interaction inside the next 30 seconds without overwriting existing comparison 70 on child 64898.','low'),
       ('moving_vehicle',65548,'Saved car boxes and sparse frames show a vehicle traversing the curved road with changing apparent position/size.','moderate'),
       ('small_distant',64911,'Saved car box about 55x29 pixels in a 2688x1520 frame; sparse frames show distant road traffic. Daylight, not a low-light claim.','moderate'),
       ('single_person_control',64006,'One person in the saved event, seven consecutive historical observations; single-person control remains limited by sparse visual review.','moderate')]
selected=[]
with sqlite3.connect('file:DEPLOYMENT/runtime/recording-index/recordings.sqlite3?mode=ro',uri=True,timeout=2) as db:
    db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
    for slot,eid,why,confidence in specs:
        item=next(x for x in media if x['event_id']==(64898 if eid==64897 else eid)).copy()
        item.update(slot=slot,event_id=eid,rationale=why,selection_confidence=confidence)
        if eid==64897:
            d=json.loads((ROOT/'private/detail-upper.json').read_text());e=next(e for e in d['events'] if e['id']==eid)
            item.update(timestamp=e['created_at'],epoch=datetime.fromisoformat(e['created_at']).timestamp())
            epoch=item['epoch'];deadline=time.monotonic()+3;db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
            segs=[dict(r) for r in db.execute("select path,start_epoch,end_epoch,playable,validated from recordings where camera_id=? and source='main' and end_epoch>? and start_epoch<? order by start_epoch limit 12",(item['camera_id'],epoch,epoch+30))]
            cursor=epoch;gaps=[]
            for s in segs:
                s['exists']=Path(s['path']).is_file()
                if not s['exists'] or not s['playable']:continue
                if s['start_epoch']>cursor+.1:gaps.append([cursor-epoch,s['start_epoch']-epoch])
                cursor=max(cursor,s['end_epoch'])
            if cursor<epoch+29.9:gaps.append([cursor-epoch,30])
            item.update(segments=segs,gaps=gaps,continuous_index_coverage=bool(segs) and not gaps,frames=[])
        assert item['continuous_index_coverage'],item
        item['finalized_evidence']='Anchor and incident end are days old; no event_tracking_jobs row exists. Historical tracker sessions are terminal/interrupted, not successful complete coverage; disclosed separately.'
        item['gap_profile']=eid==64897
        item['gap_selection_note']='Person/parked-car scene persists through the 30-second earlier-child window; modest image inspection suggests activity around gaps, to be verified against saved capture without selecting based on candidate outcomes.' if eid==64897 else None
        item['profiles']=['fixed_2fps','fixed_075fps']+(['sparse_gaps'] if item['gap_profile'] else [])
        selected.append(item)
payload={'locked_utc':datetime.now(timezone.utc).isoformat(),'candidate_sha':'67e67ad73b7dc950dd55f15cabd4cb6d8f35531d','selected':selected,'reserves':[],
         'missing':['Second suitable gap case: inspected activity generally ends before second 21; do not manufacture gap coverage.'],
         'exclusions':[{'event_id':65540,'reason':'Viewed image shows pickup occupants, not pedestrian crossing.'},{'event_id':64898,'reason':'Existing comparison row 70: preserve it. Use different actual child 64897 of same incident before corpus lock.'},{'event_id':66002,'reason':'Earlier baseline shortlist recording index had a 1-second gap.'}],
         'discovery':{'prior_event_summary_rows':1000,'prior_incident_summaries':46,'additional_api_summary_rows':200,'prior_detail_reads':46,'additional_detail_requests':11,'one_detail_request_422':'Omitted intermediate child events initially produced multiple groups; corrected with all actual child IDs.','search_windows':'72h initially; person-filtered feed widened missing close-interaction/occlusion categories to 7 days; returned oldest summary Sep 2, no 30-day search needed.'}}
path=ROOT/'corpus.json'
assert not path.exists()
path.write_text(json.dumps(payload,indent=2)+'\n')
(ROOT/'private/corpus-lock-sha256.txt').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
print(json.dumps({'locked_utc':payload['locked_utc'],'event_ids':[x['event_id'] for x in selected],'profiles':sum(len(x['profiles']) for x in selected),'missing':payload['missing']}))
