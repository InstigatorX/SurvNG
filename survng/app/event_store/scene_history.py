"""Interpret retained legacy tracking evidence without inventing detector scores."""
import math
from datetime import datetime


def legacy_track_observations(objects):
    observations = []
    for entry in objects:
        tracking = entry.get('object_tracking')
        # New tracking payloads already retain original per-frame observations.
        if not isinstance(tracking, dict) or 'scene_observations' in tracking:
            continue
        for track_index, track in enumerate(tracking.get('tracks', [])):
            if not isinstance(track, dict) or not track.get('label'):
                continue
            try:
                confidence = float(track.get('max_confidence', track.get('confidence', 0)))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence) or confidence <= 0:
                continue
            samples = track.get('box_history') or []
            if not samples and isinstance(track.get('box'), dict) and track.get('last_seen'):
                try:
                    samples = [[datetime.fromisoformat(track['last_seen']).timestamp(),
                                *(track['box'][k] for k in ('x1','y1','x2','y2'))]]
                except (KeyError, TypeError, ValueError):
                    continue
            for sample in samples:
                try:
                    at, x1, y1, x2, y2 = map(float, sample)
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(v) for v in (at,x1,y1,x2,y2)) or x2 <= x1 or y2 <= y1:
                    continue
                observations.append({
                    'label':str(track['label']), 'confidence':confidence,
                    'confidence_provenance':'track_summary',
                    'historical_track_confirmed':track.get('state') in {'confirmed','lost'},
                    'captured_at_epoch':at, 'frame_source':'legacy',
                    'frame_timestamp_exact':False, 'snapshot_visible':False,
                    'scene_track_key':track['track_id'] if track.get('track_id') is not None else f'legacy_index:{track_index}',
                    'detection_frame_width':tracking.get('frame_width'),
                    'detection_frame_height':tracking.get('frame_height'),
                    'box':dict(zip(('x1','y1','x2','y2'), (x1,y1,x2,y2))),
                    'legacy_track_state':track.get('state'),
                    'legacy_track_observations':track.get('observations'),
                })
    return observations
