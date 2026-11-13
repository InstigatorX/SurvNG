"""Record-only human reviews. Source evidence and admission are never rewritten."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time

import cv2
import numpy as np

from ..incident_utils import event_snapshot_path

MAX_ASSET_BYTES = 64 * 1024 * 1024
MAX_REVIEW_BYTES = 2 * 1024 ** 3
MAX_REVIEWS = 10000


class ReviewConflict(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class EventStoreCorrectionsMixin:
    def _init_corrections_db(self):
        self._review_lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript("""
                create table if not exists detection_review_assets (
                    id integer primary key, source text not null, source_id integer not null,
                    camera_id text not null, captured_at text not null, source_revision text not null,
                    source_state text not null, image_hash text not null, filename text not null, byte_size integer not null,
                    width integer not null, height integer not null, candidates_json text not null,
                    unique(source,source_id,source_revision)
                );
                create table if not exists detection_reviews (
                    id integer primary key, asset_id integer not null references detection_review_assets(id),
                    annotation_key text not null, observation_id text not null, verdict text not null,
                    original_label text, label text, source_kind text not null, box_json text,
                    scope text not null, location_rule integer not null, note text not null,
                    created_at real not null, created_by text not null,
                    status text not null default 'active', revision integer not null default 1,
                    idempotency_key text not null unique, request_hash text not null
                );
                create unique index if not exists detection_review_active
                    on detection_reviews(asset_id,annotation_key) where status='active';
                create index if not exists detection_review_source
                    on detection_review_assets(source,source_id);
                create table if not exists detection_review_audit (
                    id integer primary key, review_id integer not null references detection_reviews(id),
                    action text not null, actor text not null, created_at real not null,
                    details_json text not null
                );
            """)

    def _review_source(self, conn, source, source_id, observation_id=""):
        table = {"event": "events", "motion_audit": "motion_audits"}[source]
        row = conn.execute(f"select * from {table} where id=?", (source_id,)).fetchone()
        if row is None:
            raise LookupError("source evidence no longer exists")
        row = dict(row)
        objects = json.loads(row.get("objects_json") or "[]")
        candidates = []
        for index, item in enumerate(objects):
            if isinstance(item, dict) and item.get("label") and item.get("snapshot_visible") is not False:
                candidates.append({"key": f"object:{index}", "object": item})
        # Off-cover observations are offered on their own retained image only.
        supporting = []
        if source == "event":
            for item in conn.execute("select id,source_observation_id,object_index,snapshot_path,payload_json from scene_observations where event_id=? and snapshot_path!='' order by id limit 500", (source_id,)):
                supporting.append(dict(item))
        for candidate in candidates:
            linked = next((item for item in supporting
                           if f"object:{item['object_index']}" == candidate['key']
                           and json.loads(item['payload_json']).get('box') == candidate['object'].get('box')
                           and json.loads(item['payload_json']).get('label') == candidate['object'].get('label')), None)
            candidate['observation_id'] = (linked.get('source_observation_id') or linked['id']) if linked else ''
        if observation_id:
            selected = conn.execute("select id,source_observation_id,snapshot_path,payload_json from scene_observations where id=? and event_id=? and snapshot_path!=''",(observation_id,source_id)).fetchone()
            selected = dict(selected) if selected else None
            if selected and not any(item['id'] == observation_id for item in supporting):
                supporting.append(selected)
            if selected is None:
                raise LookupError("observation image is unavailable for this event")
            row['snapshot_path'] = selected['snapshot_path']
            payload = json.loads(selected['payload_json'])
            candidates = [{"key": observation_id, "object": payload, "observation_id": selected.get('source_observation_id') or observation_id}]
        revision = digest([source, source_id, row.get('evidence_revision', 0), row.get('snapshot_path'),
                           row.get('objects_json'), row.get('features_json'), row.get('created_at'), observation_id,
                           candidates])
        return row, revision, candidates, [item['id'] for item in supporting]

    def review_evidence(self, source, source_id, observation_id=""):
        with self._connect() as conn:
            row, revision, candidates, supporting = self._review_source(conn, source, source_id, observation_id)
        path = event_snapshot_path(self.storage_dir, row, self.media_storage)
        with path.open('rb') as handle:
            data = handle.read(MAX_ASSET_BYTES + 1)
        if len(data) > MAX_ASSET_BYTES:
            raise ValueError("review image exceeds 64 MiB")
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("source image cannot be decoded")
        height, width = frame.shape[:2]
        image_hash = hashlib.sha256(data).hexdigest()
        output = []
        for candidate in candidates:
            item = candidate['object']
            box = item.get('box')
            try:
                plane = [float(item['detection_frame_width']), float(item['detection_frame_height'])]
                coords = [float(box[key]) / plane[i % 2] for i, key in enumerate(('x1','y1','x2','y2'))]
                coords = self._review_box(coords)
            except (TypeError, KeyError, ValueError, ZeroDivisionError):
                continue
            output.append({'annotation_key':candidate.get('observation_id') or 'region:' + digest([item['label'],coords]), 'key': candidate['key'], 'observation_id':candidate.get('observation_id',''), 'label': str(item['label']), 'box': coords,
                           'incident_eligible': item.get('incident_eligible') is not False, 'confidence': item.get('confidence', 0), 'temporal_consensus': item.get('temporal_consensus') is True, 'semantic_tier': str(item.get('semantic_tier') or ''), 'zones': item.get('zones') or []})
        suffix = path.suffix.lower()
        if suffix not in {'.jpg', '.jpeg', '.png', '.webp'}:
            raise ValueError('unsupported source image format')
        captured_at = row['created_at']
        if observation_id and candidates:
            raw = candidates[0]['object']
            epoch = raw.get('captured_at_epoch', raw.get('frame_captured_at_epoch'))
            if isinstance(epoch, (int,float)) and math.isfinite(epoch):
                captured_at = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
        return {'source': source, 'source_id': source_id, 'camera_id': row['camera_id'],
                'captured_at': captured_at, 'observation_id': observation_id,
                'source_revision': revision, 'image_hash': image_hash,
                'revision': digest([revision, image_hash]), 'width': width, 'height': height,
                'objects': output, 'supporting_observations': supporting,
                '_data': data, '_suffix': suffix}

    @staticmethod
    def _review_box(box):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError('box must be normalized xyxy')
        values = [float(v) for v in box]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError('box must be a nonempty normalized rectangle within the image')
        return values

    def _review_directory(self):
        return self.storage_dir / 'corrections'

    @contextmanager
    def _review_asset_lock(self):
        directory = self._review_directory()
        if self._snapshot_verdict(directory / '.review.lock') not in {'present', 'confirmed_absent'}:
            raise OSError('review storage volume is unavailable')
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Serialize publication and orphan cleanup across manager/process instances.
        fd = os.open(directory / '.review.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _review_cleanup_orphans(self, conn):
        directory = self._review_directory()
        if not directory.exists():
            return
        retained = {r[0] for r in conn.execute('select filename from detection_review_assets')}
        # Only our flat, owned files; never traverse media or symlink directories.
        for path in directory.iterdir():
            if path.name not in retained and (path.name.startswith('.review-') or len(path.stem) == 64):
                if path.is_file() and not path.is_symlink() and path.stat().st_mtime < time.time() - 86400:
                    path.unlink()

    def create_detection_review(self, source, source_id, payload, actor):
        request_hash = digest([source, source_id, payload, actor])
        token = payload['idempotency_key']
        with self._review_lock, self._review_asset_lock():
            with self._connect() as conn:
                existing = conn.execute('select * from detection_reviews where idempotency_key=?', (token,)).fetchone()
                if existing:
                    if existing['request_hash'] != request_hash:
                        raise ReviewConflict('idempotency key was used for a different review')
                    return self._review_payload(conn, existing)
            evidence = self.review_evidence(source, source_id, payload.get('observation_id', ''))
            if evidence['revision'] != payload['evidence_revision']:
                raise ReviewConflict('evidence changed; reload the review image')
            verdict, source_kind = payload['verdict'], payload.get('source', 'detection')
            original = None
            raw_observation_id = evidence['observation_id']
            if verdict == 'verified_negative':
                if source != 'motion_audit':
                    raise ValueError('whole-image negative reviews require a motion audit')
                key, box, label, source_kind = 'negative', None, None, 'snapshot'
            elif source_kind == 'drawn':
                if verdict != 'missed':
                    raise ValueError('drawn boxes require a missed verdict')
                key = 'drawn:' + payload.get('annotation_id', token)
                box = self._review_box(payload.get('box'))
                label = str(payload.get('corrected_label') or '').strip()
            else:
                chosen = next((item for item in evidence['objects'] if item['key'] == payload.get('object_key')), None)
                if chosen is None:
                    raise ValueError('select an observation on this image')
                key, box, original = chosen['key'], chosen['box'], chosen['label']
                raw_observation_id = chosen['observation_id']
                key = chosen['annotation_key']
                source_kind = 'detection' if chosen['incident_eligible'] else 'hidden_candidate'
                if verdict == 'missed' and source_kind != 'hidden_candidate':
                    raise ValueError('missed requires a hidden candidate or drawn box')
                label = str(payload.get('corrected_label') or '').strip() if verdict == 'wrong_label' else original
            if label and (source_kind == 'drawn' or verdict == 'wrong_label'):
                label = ' '.join(label.split()).lower()
            if verdict != 'verified_negative' and (not label or len(label) > 100):
                raise ValueError('a label of 1 to 100 characters is required')
            if verdict not in {'not_object', 'wrong_label', 'confirmed', 'missed', 'verified_negative'}:
                raise ValueError('unknown review verdict')
            scope = payload.get('scope', 'camera')
            if scope not in {'camera', 'global'}:
                raise ValueError('invalid scope')
            location = bool(payload.get('location_rule', verdict == 'not_object'))
            filename = evidence['image_hash'] + evidence['_suffix']
            directory = self._review_directory()
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            published = directory / filename
            with self._connect() as conn:
                self._review_cleanup_orphans(conn)
                size = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
                count = conn.execute('select count(*) from detection_reviews').fetchone()[0]
                if size + (0 if published.exists() else len(evidence['_data'])) > MAX_REVIEW_BYTES or count >= MAX_REVIEWS:
                    raise ValueError('review storage quota reached (2 GiB / 10000 reviews)')
            if not published.exists():
                fd, temporary = tempfile.mkstemp(prefix='.review-', dir=directory)
                try:
                    with os.fdopen(fd, 'wb') as handle:
                        handle.write(evidence['_data'])
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, published)
                    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            # No image I/O under the shared SQLite writer. Revalidate mutable source state.
            with self._lock, self._connect() as conn:
                conn.execute('begin immediate')
                existing = conn.execute('select * from detection_reviews where idempotency_key=?', (token,)).fetchone()
                if existing:
                    if existing['request_hash'] != request_hash:
                        raise ReviewConflict('idempotency key was used for a different review')
                    return self._review_payload(conn, existing)
                if conn.execute('select count(*) from detection_reviews').fetchone()[0] >= MAX_REVIEWS:
                    raise ValueError('review count quota reached')
                if conn.execute('select coalesce(sum(byte_size),0) from (select filename,max(byte_size) as byte_size from detection_review_assets group by filename)').fetchone()[0] + (0 if conn.execute('select 1 from detection_review_assets where filename=?',(filename,)).fetchone() else len(evidence['_data'])) > MAX_REVIEW_BYTES:
                    raise ValueError('review storage quota reached')
                _, revision, _, _ = self._review_source(conn, source, source_id, evidence['observation_id'])
                if revision != evidence['source_revision']:
                    raise ReviewConflict('source changed while saving; reload the image')
                conn.execute('insert or ignore into detection_review_assets(source,source_id,camera_id,captured_at,source_revision,source_state,image_hash,filename,byte_size,width,height,candidates_json) values(?,?,?,?,?,?,?,?,?,?,?,?)',
                             (source, source_id, evidence['camera_id'], evidence['captured_at'], evidence['revision'], evidence['source_revision'], evidence['image_hash'], filename, len(evidence['_data']), evidence['width'], evidence['height'], json.dumps(evidence['objects'])))
                asset = conn.execute('select id from detection_review_assets where source=? and source_id=? and source_revision=?', (source, source_id, evidence['revision'])).fetchone()[0]
                active = conn.execute("select r.* from detection_reviews r join detection_review_assets a on a.id=r.asset_id where a.source=? and a.source_id=? and a.image_hash=? and r.status='active'", (source,source_id,evidence['image_hash'])).fetchall()
                conflicts = [r for r in active if r['annotation_key'] == key or ((r['verdict'] == 'verified_negative' and verdict in {'missed','confirmed','wrong_label'}) or (verdict == 'verified_negative' and r['verdict'] in {'missed','confirmed','wrong_label'}))]
                expected = payload.get('supersedes', {})
                if {str(r['id']): r['revision'] for r in conflicts} != expected:
                    raise ReviewConflict('active reviews conflict; explicitly supersede their current revisions')
                now = time.time()
                for old in conflicts:
                    conn.execute("update detection_reviews set status='superseded',revision=revision+1 where id=?", (old['id'],))
                    conn.execute('insert into detection_review_audit(review_id,action,actor,created_at,details_json) values(?,?,?,?,?)', (old['id'],'supersede',actor,now,json.dumps({'idempotency_key':token})))
                cursor = conn.execute('insert into detection_reviews(asset_id,annotation_key,observation_id,verdict,original_label,label,source_kind,box_json,scope,location_rule,note,created_at,created_by,idempotency_key,request_hash) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (asset,key,raw_observation_id,verdict,original,label,source_kind,json.dumps(box) if box else None,scope,int(location),payload.get('note',''),now,actor,token,request_hash))
                review_id = cursor.lastrowid
                conn.execute('insert into detection_review_audit(review_id,action,actor,created_at,details_json) values(?,?,?,?,?)', (review_id,'create',actor,now,json.dumps({'supersedes':expected})))
                return self._review_payload(conn, conn.execute('select * from detection_reviews where id=?',(review_id,)).fetchone())

    def _review_payload(self, conn, row):
        result = dict(row)
        result.pop('request_hash', None)
        result.pop('idempotency_key', None)
        raw_box = result.pop('box_json')
        result['box'] = json.loads(raw_box) if raw_box else None
        asset = dict(conn.execute('select * from detection_review_assets where id=?',(row['asset_id'],)).fetchone())
        asset.pop('filename')
        asset.pop('candidates_json')
        result['asset'] = asset
        result['match_count'] = 0
        result['last_matched_at'] = None
        return result

    def detection_reviews(self, *, before_id=None, limit=50, camera_id='', status='', verdict='', source='', source_id=None):
        clauses, values = ['1=1'], []
        for column, value in [('a.camera_id',camera_id),('r.status',status),('r.verdict',verdict),('a.source',source),('a.source_id',source_id)]:
            if value not in ('', None):
                clauses.append(column+'=?'); values.append(value)
        if before_id:
            clauses.append('r.id<?'); values.append(before_id)
        with self._connect() as conn:
            rows = conn.execute('select r.* from detection_reviews r join detection_review_assets a on a.id=r.asset_id where '+ ' and '.join(clauses)+' order by r.id desc limit ?',(*values,limit)).fetchall()
            return [self._review_payload(conn,row) for row in rows]

    def revoke_detection_review(self, review_id, expected_revision, actor):
        with self._review_lock, self._connect() as conn:
            conn.execute('begin immediate')
            row = conn.execute('select * from detection_reviews where id=?',(review_id,)).fetchone()
            if row is None:
                raise LookupError('review not found')
            if row['revision'] != expected_revision or row['status'] != 'active':
                raise ReviewConflict('review changed; refresh before undoing')
            conn.execute("update detection_reviews set status='revoked',revision=revision+1 where id=?",(review_id,))
            conn.execute('insert into detection_review_audit(review_id,action,actor,created_at,details_json) values(?,?,?,?,?)',(review_id,'revoke',actor,time.time(),'{}'))
            return self._review_payload(conn,conn.execute('select * from detection_reviews where id=?',(review_id,)).fetchone())

    def review_asset(self, asset_id):
        with self._connect() as conn:
            row = conn.execute('select * from detection_review_assets where id=?',(asset_id,)).fetchone()
            if row is None:
                raise LookupError('review image not found')
            path = self._review_directory() / row['filename']
            if self._snapshot_verdict(path) != 'present':
                raise OSError('retained review image is unavailable')
            return dict(row), path
