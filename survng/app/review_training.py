"""Training projection of direct human reviews, independent of source retention."""
import json
from pathlib import Path
from .event_store.corrections import digest


def reviewed_training_page(events, *, start, end, cameras, labels, source, kinds,
                           before, limit, base_path, reviewed_only=False, eligibility="eligible", minimum_confidence=0):
    clauses, values = ["a.captured_at>=?", "a.captured_at<?", "a.source=?",
                       "a.id=(select max(a2.id) from detection_review_assets a2 join detection_reviews r on r.asset_id=a2.id where a2.source=a.source and a2.source_id=a.source_id and a2.image_hash=a.image_hash and r.status='active')"], [start,end,source]
    if cameras:
        clauses.append('a.camera_id in ('+','.join('?' for _ in cameras)+')'); values.extend(cameras)
    if before:
        clauses.append('a.id<?'); values.append(before)
    samples, scanned, last, exhausted = [], 0, None, False
    with events._connect() as conn:
        # Bounded assets; one image may have several reviewed regions.
        rows = conn.execute('select a.* from detection_review_assets a where '+' and '.join(clauses)+' order by a.id desc limit 5000',values).fetchall()
        for asset in rows:
            scanned += 1; last = asset['id']
            reviews = conn.execute("select r.* from detection_reviews r join detection_review_assets a on a.id=r.asset_id where a.source=? and a.source_id=? and a.image_hash=? and r.status='active' order by r.id",(asset['source'],asset['source_id'],asset['image_hash'])).fetchall()
            negative = any(r['verdict']=='verified_negative' for r in reviews)
            kind = 'negative_candidate' if negative else 'annotated'
            if kinds and kind not in kinds:
                continue
            candidates = {}
            for raw in conn.execute('select candidates_json from detection_review_assets where source=? and source_id=? and image_hash=? order by id',(asset['source'],asset['source_id'],asset['image_hash'])):
                for candidate in json.loads(raw[0]):
                    candidates[candidate.get('annotation_key',candidate['key'])] = candidate
            annotations = []
            for review in reviews:
                if review['verdict']=='verified_negative':
                    continue
                if labels and str(review['label']).lower() not in labels and str(review['original_label']).lower() not in labels:
                    continue
                raw_candidate = candidates.get(review['annotation_key'], {})
                box = json.loads(review['box_json'])
                x1,y1,x2,y2 = box
                w,h = asset['width'],asset['height']
                annotations.append({
                    'annotation_id':f"review-{review['id']}", 'label':review['label'], 'confidence':None,
                    'bbox_xyxy':[x1*w,y1*h,x2*w,y2*h], 'bbox_xywh':[x1*w,y1*h,(x2-x1)*w,(y2-y1)*h],
                    'bbox_normalized_xyxy':box,'bbox_normalized_cxcywh':[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1],
                    'zones':raw_candidate.get('zones',[]), 'incident_eligible':raw_candidate.get('incident_eligible',False),
                    'temporal_consensus':raw_candidate.get('temporal_consensus',False),'semantic_tier':raw_candidate.get('semantic_tier',''),
                    'source':'survng_object_detection', 'annotation_state':'human_reviewed',
                    'review_verdict':review['verdict'], 'original_label':review['original_label'],
                    'correction_id':review['id'], 'annotation_source':'operator_drawn' if review['source_kind']=='drawn' else 'operator_reviewed',
                })
            if not reviewed_only and not negative:
                reviewed_keys = {r['annotation_key'] for r in reviews}
                for candidate in candidates.values():
                    if candidate.get('annotation_key', candidate['key']) in reviewed_keys:
                        continue
                    if labels and candidate['label'].lower() not in labels:
                        continue
                    eligible = candidate['incident_eligible']
                    if (eligibility == 'eligible' and not eligible) or (eligibility == 'ineligible' and eligible):
                        continue
                    if candidate['confidence'] < minimum_confidence:
                        continue
                    x1,y1,x2,y2 = candidate['box']
                    w,h = asset['width'],asset['height']
                    annotations.append({
                        'annotation_id':f"review-image-{asset['id']}-{candidate['key']}",
                        'label':candidate['label'], 'confidence':candidate['confidence'],
                        'bbox_xyxy':[x1*w,y1*h,x2*w,y2*h], 'bbox_xywh':[x1*w,y1*h,(x2-x1)*w,(y2-y1)*h],
                        'bbox_normalized_xyxy':candidate['box'], 'bbox_normalized_cxcywh':[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1],
                        'zones':candidate['zones'], 'incident_eligible':eligible,
                        'temporal_consensus':candidate['temporal_consensus'], 'semantic_tier':candidate['semantic_tier'],
                        'source':'survng_object_detection', 'annotation_state':'model_generated',
                    })
            if not annotations and not negative:
                continue
            if labels and negative and not annotations:
                continue
            revisions = [[r['id'],r['revision']] for r in reviews]
            samples.append({'sample_id':f"review-image-{source}-{asset['source_id']}-{asset['image_hash']}", 'revision':digest([asset['image_hash'],revisions]),
                            'event_id':asset['source_id'] if source=='event' else None, 'source_id':asset['source_id'],
                            'source':source,'sample_kind':kind,'assumed_negative':False,'camera_id':asset['camera_id'],
                            'event_at':asset['captured_at'],'captured_at':asset['captured_at'],
                            'image':{'url':f"{base_path}/api/detection-corrections/assets/{asset['id']}",
                                     'media_type':{'.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp','.png':'image/png'}[Path(asset['filename']).suffix],
                                     'width':asset['width'],'height':asset['height']},
                            'annotations':annotations,'annotation_state':'human_reviewed',
                            **({'verified_negative':True} if negative else {})})
            if len(samples) == limit:
                break
        exhausted = scanned == len(rows) and len(rows) < 5000
    return samples, scanned, None if exhausted else last, bool(rows)


def source_has_current_review(events, source, source_id):
    with events._connect() as conn:
        if conn.execute("select 1 from detection_review_assets a join detection_reviews r on r.asset_id=a.id where a.source=? and a.source_id=? and r.status='active' limit 1",(source,source_id)).fetchone() is None:
            return False
        try:
            row, revision, _, _ = events._review_source(conn,source,source_id)
        except LookupError:
            return False
        reviewed = conn.execute("select a.image_hash from detection_review_assets a join detection_reviews r on r.asset_id=a.id where a.source=? and a.source_id=? and a.source_state=? and r.status='active' limit 1",(source,source_id,revision)).fetchone()
    if not reviewed:
        return False
    from .incident_utils import event_snapshot_path
    from .event_store.corrections import MAX_ASSET_BYTES
    import hashlib
    try:
        path = event_snapshot_path(events.storage_dir,row,events.media_storage)
        with path.open('rb') as handle:
            data = handle.read(MAX_ASSET_BYTES+1)
    except (FileNotFoundError,PermissionError):
        return False
    return hashlib.sha256(data).hexdigest() == reviewed[0]
