"""Optional retained gallery images. They never participate in cover/admission policy."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone

from ..evidence_gallery import MAX_DETECTION_IMAGES, MAX_GALLERY_IMAGES, MAX_IMAGE_BYTES, TRACKING_SOURCE


class EventStoreGalleryMixin:
    def _init_gallery_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript('''
                create table if not exists scene_evidence_images (
                    id text primary key,
                    event_id integer not null references events(id) on delete cascade,
                    frame_key text not null,
                    captured_epoch real not null,
                    snapshot_path text not null default '',
                    payload_json text not null,
                    state text not null,
                    lease_token text not null default '',
                    lease_until real not null default 0
                );
                create index if not exists scene_evidence_event on scene_evidence_images(event_id);
                create index if not exists scene_evidence_snapshot on scene_evidence_images(snapshot_path);
            ''')

    @staticmethod
    def _gallery_incident(conn, event_id):
        row = conn.execute('select p.incident_id from scene_event_membership m '
            'join scene_episodes p on p.id=m.episode_id where m.event_id=?', (event_id,)).fetchone()
        return row[0] if row else None

    @staticmethod
    def _gallery_rows(conn, incident_id):
        return conn.execute('select g.* from scene_evidence_images g '
            'join scene_event_membership m on m.event_id=g.event_id '
            'join scene_episodes p on p.id=m.episode_id where p.incident_id=? '
            'order by g.captured_epoch,g.id', (incident_id,)).fetchall()

    @staticmethod
    def _gallery_remaining(payloads, source):
        """Free slots for one source. Detection may not use tracking's share."""
        remaining = MAX_GALLERY_IMAGES - len(payloads)
        if source != TRACKING_SOURCE:
            detection = sum(payload.get('source') != TRACKING_SOURCE for payload in payloads)
            remaining = min(remaining, MAX_DETECTION_IMAGES - detection)
        return max(0, remaining)

    def reserve_scene_evidence_slots(self, event_id, count, *, source='detection'):
        """Occupy slots before a caller encodes, so tracking cannot take them.

        The returned pending rows are the caller's capacity. Success converts
        those rows; failure, cancellation, and expiry release only the unused
        ones. Tracking may use whatever this reservation does not hold.
        """
        count = max(0, int(count or 0))
        if count <= 0 or event_id is None:
            return None
        now, token = time.time(), uuid.uuid4().hex
        ids = []
        with self._lock, self._connect() as conn:
            conn.execute('begin immediate')
            incident = self._gallery_incident(conn, event_id)
            if not incident:
                return None
            rows = self._gallery_rows(conn, incident)
            for row in rows:
                if row['state'] == 'pending' and row['lease_until'] <= now:
                    conn.execute('delete from scene_evidence_images where id=?', (row['id'],))
            rows = [row for row in rows if row['state'] != 'pending' or row['lease_until'] > now]
            payloads = [json.loads(row['payload_json']) for row in rows]
            for index in range(count):
                if self._gallery_remaining(payloads, source) <= 0:
                    break
                image_id = 'gallery-' + uuid.uuid4().hex
                payload = {'source': source, 'reservation': True, 'camera_id': ''}
                conn.execute(
                    'insert into scene_evidence_images '
                    '(id,event_id,frame_key,captured_epoch,payload_json,state,lease_token,lease_until) '
                    'values(?,?,?,?,?,\'pending\',?,?)',
                    (image_id, event_id, f'reservation:{token}:{index}', now, json.dumps(payload), token, now + 300),
                )
                payloads.append(payload)
                ids.append(image_id)
        if not ids:
            return None
        return {'token': token, 'ids': ids, 'event_id': int(event_id)}

    def release_scene_evidence_reservation(self, reservation) -> None:
        if not reservation:
            return
        ids = list(reservation.get('ids') or [])
        token = str(reservation.get('token') or '')
        if not ids or not token:
            return
        with self._lock, self._connect() as conn:
            conn.execute('begin immediate')
            conn.execute(
                f"delete from scene_evidence_images where lease_token=? and state='pending' "
                f"and id in ({','.join('?' for _ in ids)})",
                (token, *ids),
            )

    def finish_scene_evidence_reservation(self, event_id, reservation, images, snapshot_writer) -> int:
        """Turn reserved rows into retained images and release the unused tail."""
        if not reservation:
            return self.retain_scene_evidence_images(event_id, images, snapshot_writer)
        ids = list(reservation.get('ids') or [])
        token = str(reservation.get('token') or '')
        saved = 0
        used: list[str] = []
        for image, image_id in zip(images, ids):
            if len(image.image.data) > MAX_IMAGE_BYTES:
                continue
            path = ''
            adopted = False
            try:
                path = snapshot_writer(image.image, datetime.fromtimestamp(image.captured_epoch, timezone.utc))
                with self._lock, self._connect() as conn:
                    conn.execute('begin immediate')
                    retained = self._acquired_media_path(conn, path) if path else ''
                    updated = False
                    if retained:
                        updated = bool(conn.execute(
                            "update scene_evidence_images set snapshot_path=?, captured_epoch=?, "
                            "payload_json=?, state='ready', lease_until=0 "
                            "where id=? and lease_token=? and state='pending'",
                            (retained, image.captured_epoch, json.dumps(image.metadata), image_id, token),
                        ).rowcount)
                    if updated:
                        self._register_acquired_snapshot(
                            conn, retained, image.metadata['camera_id'],
                            image.captured_epoch, len(image.image.data),
                        )
                        conn.execute(
                            'update events set scene_media_revision=scene_media_revision+1 where id=?',
                            (event_id,),
                        )
                        adopted = True
                    else:
                        conn.execute(
                            "delete from scene_evidence_images where id=? and lease_token=? and state='pending'",
                            (image_id, token),
                        )
                if adopted:
                    saved += 1
                    used.append(image_id)
            finally:
                if not adopted and path:
                    self._delete_snapshot_if_unreferenced(path)
        unused = [image_id for image_id in ids if image_id not in used]
        if unused:
            self.release_scene_evidence_reservation({'token': token, 'ids': unused})
        return saved

    def scene_evidence_capacity(self, event_id=None, *, source='detection'):
        if event_id is None:
            return {'remaining':self._gallery_remaining([], source), 'frames':[]}
        with self._connect() as conn:
            incident = self._gallery_incident(conn, event_id)
            if not incident:
                return {'remaining':0, 'frames':[]}
            rows = self._gallery_rows(conn, incident)
            retained = [json.loads(r['payload_json']) for r in rows
                        if r['state'] != 'pending' or r['lease_until'] > time.time()]
            return {'remaining':self._gallery_remaining(retained, source), 'frames':retained}

    def retain_scene_evidence_images(self, event_id, images, snapshot_writer):
        """Reserve a durable slot before I/O; finish only our own reservation.

        A crashed writer's slot can be reclaimed after its lease. Ready/expired
        slots are never replenished by refinement retries. No admission is run.
        """
        saved = 0
        for image in images[:MAX_GALLERY_IMAGES]:
            if len(image.image.data) > MAX_IMAGE_BYTES:
                continue
            now, token = time.time(), uuid.uuid4().hex
            with self._lock, self._connect() as conn:
                conn.execute('begin immediate')
                incident = self._gallery_incident(conn, event_id)
                if not incident:
                    break
                rows = self._gallery_rows(conn, incident)
                for row in rows:
                    if row['state'] == 'pending' and row['lease_until'] <= now:
                        conn.execute('delete from scene_evidence_images where id=?', (row['id'],))
                rows = [r for r in rows if r['state'] != 'pending' or r['lease_until'] > now]
                if not self._gallery_remaining([json.loads(r['payload_json']) for r in rows],
                                               image.metadata.get('source')):
                    break
                if any(r['frame_key'] == image.key for r in rows):
                    continue
                image_id = 'gallery-' + hashlib.sha256(f'{incident}|{image.key}'.encode()).hexdigest()[:32]
                conn.execute('insert into scene_evidence_images '
                    '(id,event_id,frame_key,captured_epoch,payload_json,state,lease_token,lease_until) '
                    'values(?,?,?,?,?,\'pending\',?,?)',
                    (image_id,event_id,image.key,image.captured_epoch,json.dumps(image.metadata),token,now+60))
            path = ''
            adopted = False
            try:
                path = snapshot_writer(image.image, datetime.fromtimestamp(image.captured_epoch, timezone.utc))
                with self._lock, self._connect() as conn:
                    conn.execute('begin immediate')
                    retained = self._acquired_media_path(conn, path) if path else ''
                    if retained:
                        updated = bool(conn.execute('update scene_evidence_images set snapshot_path=?,state=\'ready\',lease_until=0 '
                            'where id=? and lease_token=? and state=\'pending\'', (retained,image_id,token)).rowcount)
                    else:
                        updated = False
                    if updated:
                        self._register_acquired_snapshot(conn, retained, image.metadata['camera_id'],
                            image.captured_epoch, len(image.image.data))
                        conn.execute('update events set scene_media_revision=scene_media_revision+1 where id=?', (event_id,))
                        row = conn.execute('select * from events where id=?', (event_id,)).fetchone()
                        self._evidence_outbox(conn, row, 'gallery_updated', reason='gallery_image_retained', notify=False)
                        current_incident = self._gallery_incident(conn, event_id)
                        if current_incident:
                            self._scene_changed(conn, current_incident, notify=False)
                    else:
                        conn.execute('delete from scene_evidence_images where id=? and lease_token=? and state=\'pending\'', (image_id,token))
                adopted = updated
                if adopted:
                    saved += 1
            finally:
                if not adopted:
                    # Also release reservations after an exceptional writer failure.
                    with self._lock, self._connect() as conn:
                        conn.execute('delete from scene_evidence_images where id=? and lease_token=? and state=\'pending\'', (image_id,token))
                    if path:
                        self._delete_snapshot_if_unreferenced(path)
        return saved

    def scene_evidence_image(self, image_id):
        with self._connect() as conn:
            row = conn.execute('select * from scene_evidence_images where id=? and state=\'ready\'', (image_id,)).fetchone()
            return dict(row) if row else None

    def _scene_gallery_payload(self, conn, incident_id):
        images = []
        for row in self._gallery_rows(conn, incident_id):
            if row['state'] != 'ready' or not row['snapshot_path']:
                continue
            metadata = json.loads(row['payload_json'])
            objects = [{**o, 'snapshot_visible':True} for o in metadata.get('objects', [])]
            images.append({'id':row['id'], 'event_id':row['event_id'], 'camera_id':metadata['camera_id'],
                'captured_at':datetime.fromtimestamp(row['captured_epoch'], timezone.utc).isoformat(),
                'captured_epoch':row['captured_epoch'], 'width':metadata['width'], 'height':metadata['height'],
                'frame_timestamp_exact':metadata.get('frame_timestamp_exact', False),
                'snapshot_url':f'/api/incidents/evidence/{row["id"]}/snapshot', 'objects':objects,
                'role':'gallery'})
        return images[:MAX_GALLERY_IMAGES]
