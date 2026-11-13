"""Authenticated, generation-leased review APIs. No inference or admission hooks."""
from __future__ import annotations

from contextlib import contextmanager
import json
from typing import Literal

import cv2
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .event_store.corrections import ReviewConflict, MAX_REVIEW_BYTES, MAX_REVIEWS
from .manager_access import manager_generation_lease


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    evidence_revision: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=8, max_length=100)
    verdict: Literal['not_object','wrong_label','confirmed','missed','verified_negative']
    source: Literal['detection','hidden_candidate','drawn'] = 'detection'
    observation_id: str = Field(default='', max_length=200)
    object_key: str = Field(default='', max_length=200)
    annotation_id: str = Field(default='', max_length=100)
    corrected_label: str | None = Field(default=None, max_length=100)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    scope: Literal['camera','global'] = 'camera'
    location_rule: bool | None = None
    note: str = Field(default='', max_length=2000)
    supersedes: dict[str, int] = Field(default_factory=dict, max_length=100)


def create_correction_router(deps):
    router = APIRouter()

    def actor(request):
        principal = request.scope.get('survng_principal')
        if principal is not None:
            if not principal.permits('admin'):
                raise HTTPException(403, 'administrator access required')
            return f'{principal.kind}:{principal.token_id}'
        config = deps.get_config()
        if config.api_auth.enabled or config.web_auth.enabled:
            raise HTTPException(401, 'authentication required')
        return 'local:authentication-disabled'

    def can_review(request):
        try:
            actor(request)
            return True
        except HTTPException:
            return False

    @contextmanager
    def store():
        try:
            with manager_generation_lease(deps.manager_access, deps.manager_lock, deps.get_manager) as manager:
                yield manager.events
        except ReviewConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except (LookupError, FileNotFoundError) as exc:
            raise HTTPException(404, 'review evidence is unavailable') from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(503, 'review storage is unavailable; no review was confirmed') from exc

    def evidence(source, source_id, observation_id, request):
        with store() as events:
            result = events.review_evidence(source, source_id, observation_id)
            result = {k:v for k,v in result.items() if not k.startswith('_')}
            result['can_review'] = can_review(request)
            return result

    @router.get('/api/events/{event_id}/review-evidence')
    def event_evidence(event_id: int, request: Request, observation_id: str = ''):
        return evidence('event', event_id, observation_id, request)

    @router.get('/api/motion-audits/{audit_id}/review-evidence')
    def audit_evidence(audit_id: int, request: Request):
        return evidence('motion_audit', audit_id, '', request)

    @router.get('/api/detection-corrections/source-image')
    def source_image(source: Literal['event','motion_audit'], source_id: int,
                     revision: str, observation_id: str = ''):
        with store() as events:
            evidence = events.review_evidence(source, source_id, observation_id)
            if evidence['revision'] != revision:
                raise ReviewConflict('image changed; refresh before reviewing')
            return Response(evidence['_data'], media_type=_media_type(evidence['_suffix']), headers={'Cache-Control':'private, no-store'})

    def create(source, source_id, payload, request):
        reviewer = actor(request)
        values = payload.model_dump(exclude_none=True)
        if not values.get('annotation_id'):
            values.pop('annotation_id', None)
        with store() as events:
            return events.create_detection_review(source, source_id, values, reviewer)

    @router.post('/api/events/{event_id}/corrections')
    def event_review(event_id: int, payload: ReviewRequest, request: Request):
        if payload.verdict == 'verified_negative':
            raise HTTPException(422, 'whole-image negatives require a motion audit')
        return create('event', event_id, payload, request)

    @router.post('/api/motion-audits/{audit_id}/corrections')
    def audit_review(audit_id: int, payload: ReviewRequest, request: Request):
        return create('motion_audit', audit_id, payload, request)

    @router.put('/api/motion-audits/{audit_id}/review')
    def audit_negative(audit_id: int, payload: ReviewRequest, request: Request):
        if payload.verdict != 'verified_negative':
            raise HTTPException(422, 'expected verified_negative')
        return create('motion_audit', audit_id, payload, request)

    @router.get('/api/detection-corrections')
    def list_reviews(camera_id: str = '', status: Literal['','active','revoked','superseded'] = '',
                     verdict: str = '', source: Literal['','event','motion_audit'] = '',
                     source_id: int | None = None, cursor: int | None = Query(None, gt=0),
                     limit: int = Query(50, ge=1, le=100)):
        with store() as events:
            rows = events.detection_reviews(before_id=cursor, limit=limit+1, camera_id=camera_id,
                                             status=status, verdict=verdict, source=source, source_id=source_id)
            return {'items':rows[:limit], 'next_cursor':rows[limit-1]['id'] if len(rows)>limit else None}

    @router.delete('/api/detection-corrections/{review_id}')
    def revoke(review_id: int, request: Request, revision: int = Query(..., ge=1)):
        reviewer = actor(request)
        with store() as events:
            return events.revoke_detection_review(review_id, revision, reviewer)

    @router.get('/api/detection-corrections/status')
    def review_status():
        with store() as events, events._connect() as conn:
            return {'mode':'off', 'embedding':'deferred', 'runtime_matching':False,
                    'counts':dict(conn.execute('select status,count(*) from detection_reviews group by status').fetchall()),
                    'bytes':conn.execute('select coalesce(sum(byte_size),0) from (select filename,max(byte_size) as byte_size from detection_review_assets group by filename)').fetchone()[0],
                    'quota_bytes':MAX_REVIEW_BYTES, 'max_reviews':MAX_REVIEWS}

    @router.get('/api/detection-corrections/assets/{asset_id}')
    def asset_image(asset_id: int, review_id: int | None = None):
        with store() as events:
            asset, path = events.review_asset(asset_id)
            if not path.is_file():
                raise FileNotFoundError(path)
            if review_id is not None:
                with events._connect() as conn:
                    row = conn.execute('select box_json from detection_reviews where id=? and asset_id=?',(review_id,asset_id)).fetchone()
                if row is None:
                    raise LookupError('review not found')
                if row[0]:
                    frame = cv2.imread(str(path))
                    if frame is None:
                        raise ValueError('retained image cannot be decoded')
                    x1,y1,x2,y2 = json.loads(row[0])
                    h,w = frame.shape[:2]
                    crop = frame[max(0,int(y1*h)):min(h,max(int(y1*h)+1,int(y2*h))), max(0,int(x1*w)):min(w,max(int(x1*w)+1,int(x2*w)))]
                    ok, encoded = cv2.imencode('.jpg',crop)
                    if not ok:
                        raise ValueError('crop cannot be encoded')
                    return Response(encoded.tobytes(),media_type='image/jpeg',headers={'Cache-Control':'private, no-store'})
            return FileResponse(path, media_type=_media_type(path.suffix), headers={'Cache-Control':'private, no-store'})

    @router.get('/api/detection-corrections/review-queue')
    def queue(cursor: str = '', limit: int = Query(25,ge=1,le=50), camera_id: str = ''):
        # Priority plus ID form a stable keyset; no unbounded image decode in listing.
        try:
            before_priority, before_id = [int(v) for v in cursor.split(':')] if cursor else (2, 2**63-1)
            if before_priority not in (0,1,2) or before_id < 1:
                raise ValueError()
        except ValueError as exc:
            raise HTTPException(422,'invalid review queue cursor') from exc
        with store() as events, events._connect() as conn:
            # Use the latest completed camera review, not raw motion telemetry.
            # AI hints only rank images; they never create human annotations.
            priority = """case when exists (
                select 1 from json_each(coalesce((
                    select result_json from motion_ai_reviews ai
                    where ai.camera_id=motion_audits.camera_id and ai.status='completed'
                    order by ai.created_at desc,ai.id desc limit 1
                ), '{}'), '$.samples') sample
                where json_extract(sample.value,'$.kind')='motion_decision'
                and json_extract(sample.value,'$.audit_id')=motion_audits.id
                and json_extract(sample.value,'$.verdict')='likely_miss'
            ) then 1 else 0 end"""
            rows = conn.execute(f"select *, {priority} as priority from motion_audits where snapshot_path!='' and (?='' or camera_id=?) and ({priority}<? or ({priority}=? and id<?)) order by priority desc,id desc limit 200",
                                (camera_id,camera_id,before_priority,before_priority,before_id)).fetchall()
            result, last = [], None
            for row in rows:
                last = row
                _, revision, _, _ = events._review_source(conn,'motion_audit',row['id'])
                reviewed = conn.execute("select 1 from detection_review_assets a join detection_reviews r on r.asset_id=a.id where a.source='motion_audit' and a.source_id=? and a.source_state=? and r.status='active' limit 1",(row['id'],revision)).fetchone()
                if reviewed:
                    continue
                result.append({'id':row['id'],'camera_id':row['camera_id'],'created_at':row['created_at'], 'probable_miss':bool(row['priority'])})
                if len(result) == limit:
                    break
            return {'items':result,'next_cursor':f"{last['priority']}:{last['id']}" if last is not None and (len(result)==limit or len(rows)==200) else None}

    @router.get('/api/detection-corrections/{review_id}')
    def review_detail(review_id: int):
        with store() as events, events._connect() as conn:
            row = conn.execute('select * from detection_reviews where id=?',(review_id,)).fetchone()
            if row is None:
                raise LookupError('review not found')
            result = events._review_payload(conn,row)
            result['audit'] = [dict(item) for item in conn.execute('select action,actor,created_at,details_json from detection_review_audit where review_id=? order by id',(review_id,))]
            return result

    return router


def _media_type(suffix):
    return {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp','.png':'image/png'}[suffix]
