"""Small evidence analyzer: use exact shared inputs and existing whole-track assignment.
Does not produce identity labels, IDF1, or accuracy estimates.
"""
import json,csv,sys,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'checkout'))
from survng.app.object_track.assignment import maximum_weight_assignment
from survng.app.object_track.geometry import _box,_iou
from survng.app.tracking_evaluation import selected_frames
B='survng_hybrid';C='survng_hybrid_multicue'
def read(path):return json.loads(path.read_text())
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def equivalent(a,b):
    return a['label']==b['label'] and all(abs(x-y)<=1e-5 for x,y in zip(_box(a['box']),_box(b['box'])))
def align(result,replay,profile):
    frames=selected_frames(replay,profile);members={};ambiguous=[];unmatched=[]
    for engine in (B,C):
        mapping={}
        for frame,outputs in zip(frames,result['engines'][engine]['frame_observations'],strict=True):
            assert frame['frame_index']==outputs['frame_index'] and frame['captured_at']==outputs['captured_at']
            detections=frame['detections'];bad=set()
            for i,d in enumerate(detections):
                for j in range(i):
                    if d['label']==detections[j]['label'] and _iou(_box(d['box']),_box(detections[j]['box']))>=.95:bad.update((i,j))
            for out in outputs['objects']:
                matches=[i for i,d in enumerate(detections) if equivalent(out,d)]
                if len(matches)!=1 or matches[0] in bad:
                    (ambiguous if matches else unmatched).append({'engine':engine,'frame_index':frame['frame_index'],'track_id':out['track_id'],'matching_input_indexes':matches});continue
                key=(frame['frame_index'],matches[0])
                if key in mapping:raise ValueError('multiple confirmed outputs claimed one input observation')
                mapping[key]=out['track_id']
        members[engine]=mapping
    b,c=members[B],members[C];bids=sorted(set(b.values()));cids=sorted(set(c.values()));edges=collections.Counter((b[k],c[k]) for k in b.keys()&c.keys())
    weights=[[edges[x,y] for y in cids] for x in bids]
    matches=maximum_weight_assignment(weights)
    whole={cids[j]:bids[i] for i,j in matches}
    ties=[bids[i] for i,row in enumerate(weights) if row and max(row)>0 and row.count(max(row))>1]
    changes=[];timestamps={f['frame_index']:f['captured_at'] for f in frames}
    for key in sorted(b.keys()|c.keys()):
        left=b.get(key);right=c.get(key)
        if left is not None and right is not None and whole.get(right)==left:continue
        changes.append({'frame_index':key[0],'detection_index':key[1],'captured_at':timestamps[key[0]],'baseline_track_id':left,'candidate_track_id':right,
                        'kind':'confirmed_presence_difference' if left is None or right is None else 'whole_track_continuity_disagreement'})
    b_to_c={x:sorted(y for xx,y in edges if xx==x) for x in bids};c_to_b={y:sorted(x for x,yy in edges if yy==y) for y in cids}
    cue=result['engines'][C].get('reid_diagnostics',{}).get('association_cues',{})
    return {'method':'Whole-track maximum-weight one-to-one alignment using shared exact input observation membership, never per-frame ID remapping. IoU>=0.95 same-class overlapping input boxes are ambiguous/excluded from identity membership. This is tracker agreement, NOT identity ground truth.',
            'candidate_to_baseline_whole_track_map':whole,'alignment_row_max_ties':ties,'overlap_edges':[{'baseline':x,'candidate':y,'shared_observations':n} for (x,y),n in sorted(edges.items())],
            'split_disagreements':{x:ys for x,ys in b_to_c.items() if len(ys)>1},'merge_disagreements':{y:xs for y,xs in c_to_b.items() if len(xs)>1},
            'ambiguous_observations':ambiguous,'unmatched_output_observations':unmatched,'changes':changes,'changed_frames':sorted({c['frame_index'] for c in changes}),
            'zero_adjustment_disagreement':bool(changes) and cue.get('counts',{}).get('adjusted_pairs',0)==0}
def main():
    corpus=read(ROOT/'corpus.json');rows=[];episodes=[]
    for case in corpus['selected']:
        replay_path=ROOT/'private'/f'capture-{case["event_id"]}.replay.json'
        if not replay_path.exists():continue
        replay=read(replay_path)
        for profile in case['profiles']:
            path=ROOT/'results'/f'{case["event_id"]}-{profile}.json'
            if not path.exists():continue
            result=read(path);engines=result['engines'];row={k:case[k] for k in ('event_id','camera_id','incident_id','slot')};row.update(profile=profile,frames=result['frames_processed'],effective_fps=result['effective_sample_fps'],status='PASS',engine_errors='')
            errors={k:v['error'] for k,v in engines.items() if v.get('error')}
            if errors:row.update(status='ENGINE_ERROR',engine_errors=json.dumps(errors));rows.append(row);continue
            b,c=engines[B],engines[C];counts=c['reid_diagnostics']['association_cues']['counts']
            for prefix,e in [('baseline',b),('candidate',c)]:
                for dst,src in [('confirmed_track_count','track_count'),('confirmed_observations','observations'),('fragmentation_proxy','fragmentation_proxy'),('reid_recoveries','reid_recoveries'),('supplied_embeddings','appearance_input_count'),('average_ms_per_frame','average_ms_per_frame')]:row[prefix+'_'+dst]=e[src]
                row[prefix+'_new_track_count']=e['reid_diagnostics']['association_counts']['new_track']
            row.update(absolute_delta_ms=round(c['average_ms_per_frame']-b['average_ms_per_frame'],6),candidate_baseline_ratio=c['average_ms_per_frame']/b['average_ms_per_frame'] if b['average_ms_per_frame'] else None,
                       valid_cue_pairs=counts['valid_pairs'],ambiguous_cue_pairs=counts['ambiguous_pairs'],adjusted_cue_pairs=counts['adjusted_pairs'],direction_available=counts['direction_available'],confidence_available=counts['confidence_available'])
            a=align(result,replay,profile);save(ROOT/'results'/f'{case["event_id"]}-{profile}.agreement.json',a)
            row.update(changed_observations=len(a['changes']),changed_frames=len(a['changed_frames']),ambiguous_observations=len(a['ambiguous_observations']),unmatched_output_observations=len(a['unmatched_output_observations']),zero_adjustment_disagreement=a['zero_adjustment_disagreement'])
            groups=[]
            for change in a['changes']:
                if not groups or change['captured_at']-groups[-1][-1]['captured_at']>2:groups.append([])
                groups[-1].append(change)
            cue=c['reid_diagnostics']['association_cues']
            for group in groups:
                times={x['captured_at'] for x in group};samples=[x for x in cue['pair_samples'] if any(abs(x['captured_at']-t)<1e-4 for t in times)]
                episodes.append({'event_id':case['event_id'],'camera_id':case['camera_id'],'profile':profile,'first_epoch':group[0]['captured_at'],'last_epoch':group[-1]['captured_at'],'changes':group,'retained_cue_pairs_at_changed_times':samples,'pair_samples_truncated':cue['pair_samples_truncated'],'note':'Evaluated cue pairs are not necessarily selected assignments.'})
            rows.append(row)
    episodes.sort(key=lambda e:(e['event_id'],e['profile'],e['first_epoch']))
    save(ROOT/'review/episodes.json',episodes[:3])
    summary={'rows':rows,'completed_paired_evaluations':sum(r['status']=='PASS' for r in rows),'expected_paired_evaluations':14,'planned_paired_evaluations':13,
             'case_profiles_scores_affected':sum(r.get('adjusted_cue_pairs',0)>0 for r in rows),'distinct_incidents_scores_affected':len({r['event_id'] for r in rows if r.get('adjusted_cue_pairs',0)>0}),
             'case_profiles_with_disagreements':sum(r.get('changed_frames',0)>0 for r in rows),'material_disagreement_episodes':len(episodes),'investigation_episode_selection':'First three sorted by event ID, profile name, first timestamp. Group changed observations within 2 seconds.',
             'idf1':'NOT MEASURED','true_identity_switches':'NOT MEASURED','true_false_merges':'NOT MEASURED','missing':corpus['missing']}
    save(ROOT/'summary.json',summary)
    with (ROOT/'summary.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
    for r in rows:print(r['event_id'],r['profile'],'tracks',r.get('baseline_confirmed_track_count'),r.get('candidate_confirmed_track_count'),'cueadj',r.get('adjusted_cue_pairs'),'changed',r.get('changed_frames'),'ms',r.get('baseline_average_ms_per_frame'),r.get('candidate_average_ms_per_frame'))
if __name__=='__main__':main()
