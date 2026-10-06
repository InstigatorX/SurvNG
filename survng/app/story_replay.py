"""Evidence-bounded automatic camera direction and replay planning."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone


def epoch(value):
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        try:
            result = datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            return None
    return result if math.isfinite(result) else None


def evidence_fingerprint(incidents):
    return hashlib.sha256(json.dumps([(i["id"], i.get("revision"), [(e.get("id"), e.get("evidence_revision"), e.get("scene_media_revision")) for e in i.get("events", [])]) for i in incidents], sort_keys=True).encode()).hexdigest()


def crop_path(incident, camera_id, start, end, subject_ids=()):
    """Union simultaneous subjects; never crop to an arbitrary first detection."""
    samples = {}
    for subject in incident.get("scene_objects", []):
        if subject_ids and subject["id"] not in subject_ids:
            continue
        for o in subject.get("observations", [])[:2000]:
            at = epoch(o.get("captured_at"))
            w, h, box = o.get("detection_frame_width", 0), o.get("detection_frame_height", 0), o.get("box")
            if o.get("camera_id") != camera_id or at is None or not start <= at <= end or not box or not w or not h:
                continue
            values = [box.get(k) for k in ("x1", "y1", "x2", "y2")]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) or values[2] <= values[0] or values[3] <= values[1]:
                continue
            b = [max(0, min(1, v / (w if n % 2 == 0 else h))) for n, v in enumerate(values)]
            samples.setdefault(round(at, 1), []).append(b)
    points = []
    for at, boxes in sorted(samples.items()):
        x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        # Conservative padding and 2x digital zoom cap.
        size = min(1, max(.5, (x2-x1)*1.5, (y2-y1)*1.5))
        points.append({"at": at-start, "x": (x1+x2)/2, "y": (y1+y2)/2, "size": size})
    if len(points) < 2:
        return []  # Single frames cannot direct motion.
    # Sparse observations cannot justify following a subject through blind time.
    if any(b["at"] - a["at"] > 5 for a, b in zip(points, points[1:])):
        return []
    step = max(1, math.ceil(len(points)/40))
    retained = points[::step]
    if retained[-1] != points[-1]:
        retained.append(points[-1])
    return retained


def crop_at(points, at):
    """Smoothstep interpolation shared with the browser/FFmpeg implementations."""
    if not points or at < points[0]["at"] or at > points[-1]["at"]:
        return {"x": .5, "y": .5, "size": 1}
    left = points[0]
    for right in points[1:]:
        if at <= right["at"]:
            u = max(0, min(1, (at-left["at"])/max(.001, right["at"]-left["at"])))
            u = u*u*(3-2*u)
            p = {k: left[k]+(right[k]-left[k])*u for k in ("x", "y", "size")}
            # Establishing wide view, then ease into the crop.
            intro = min(1, max(0, (at-1)/2))
            p["size"] = 1+(p["size"]-1)*intro
            half = p["size"]/2
            p["x"] = max(half, min(1-half, p["x"]))
            p["y"] = max(half, min(1-half, p["y"]))
            return p
        left = right
    return {"x": .5, "y": .5, "size": 1}


def build_replay_plan(incidents, recording_rows, *, directed=True, split_screen=True, padding=2, member_subjects=None, order="chronological"):
    if order == "member" and len(incidents) > 1:
        output, missing, duration, previous = [], [], 0, None
        for incident in incidents:
            part = build_replay_plan([incident], recording_rows, directed=directed, split_screen=split_screen, padding=padding, member_subjects=member_subjects)
            videos = [shot for shot in part["shots"] if shot["kind"] == "video"]
            if videos and previous is not None and abs(videos[0]["start"]-previous) > .01:
                output.append({"kind": "gap", "duration": 2, "elapsed_seconds": videos[0]["start"]-previous, "start": previous, "end": videos[0]["start"], "offset": duration})
                duration += 2
            for shot in part["shots"]:
                output.append(shot | {"offset": duration + shot["offset"]})
            duration += part["duration"]
            missing.extend(part["missing_coverage"])
            if videos:
                previous = videos[-1]["end"]
        if len(output)>512 or duration>4200:
            raise ValueError("Replay exceeds the rendering limit; split this Storyline")
        return {"shots": output, "duration": duration, "missing_coverage": missing[:256], "missing_coverage_truncated": len(missing)>256, "directed": directed, "split_screen": split_screen, "order": order, "audio": "muted", "evidence_fingerprint": evidence_fingerprint(incidents)}
    candidates, missing = [], []
    for incident in incidents:
        for episode in incident.get("episodes") or [{"id": incident["id"], "camera_id": incident.get("camera_id"), "start_at": incident.get("start_at"), "end_at": incident.get("end_at")}]:
            start, end = epoch(episode.get("start_at")), epoch(episode.get("end_at"))
            if start is None or end is None or not episode.get("camera_id"):
                continue
            start, end = start-padding, max(start+padding+1, end)+padding
            if end-start > 3600:
                raise ValueError("An episode exceeds the one-hour replay limit; split its incident first")
            rows = recording_rows(episode["camera_id"], start, end)
            spans = sorted([(max(start, float(r["start_epoch"])), min(end, float(r["end_epoch"])), str(r.get("stream_fingerprint") or "")) for r in rows if float(r["end_epoch"])>start and float(r["start_epoch"])<end])
            merged = []
            for a, b, fingerprint in spans:
                if b <= a:
                    continue
                if merged and a <= merged[-1][1] and fingerprint == merged[-1][2]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b, fingerprint])
            cursor = start
            for a, b, _fingerprint in merged:
                if a > cursor:
                    missing.append({"camera_id": episode["camera_id"], "start": cursor, "end": a})
                cursor = b
                points = crop_path(incident, episode["camera_id"], a, b, (member_subjects or {}).get(incident["id"], [])) if directed else []
                candidates.append({"incident_id": incident["id"], "episode_id": episode["id"], "camera_id": episode["camera_id"], "start": a, "end": b, "crop": points, "stream_fingerprint": _fingerprint, "score": len(points), "relation": "selected_incident"})
            if cursor < end:
                missing.append({"camera_id": episode["camera_id"], "start": cursor, "end": end})
    if candidates and max(c["end"] for c in candidates)-min(c["start"] for c in candidates)>21600:
        raise ValueError("A replay supports at most six hours of source time; split this Storyline")
    boundaries = sorted({v for c in candidates for v in (c["start"], c["end"])})
    shots = []
    for a, b in zip(boundaries, boundaries[1:]):
        if b-a < .01:
            continue
        available = [c for c in candidates if c["start"] <= a and c["end"] >= b]
        by_camera = {}
        for c in sorted(available, key=lambda c: (-c["score"], c["camera_id"], c["incident_id"])):
            by_camera.setdefault(c["camera_id"], c)
        selected = list(by_camera.values())[:2 if split_screen else 1]
        if not selected:
            continue
        views = []
        for c in selected:
            # Points remain in their original recording-span time coordinates.
            views.append({k: c[k] for k in ("incident_id", "episode_id", "camera_id", "crop", "stream_fingerprint")} | {"crop_offset": a-c["start"]})
        if shots and shots[-1]["end"] == a and shots[-1]["views"] == views:
            shots[-1]["end"] = b
        else:
            shots.append({"start": a, "end": b, "views": views})
    if len(shots) > 256 or sum(s["end"]-s["start"] for s in shots)>3600:
        raise ValueError("Replay exceeds 256 shots or one hour; split this Storyline")
    cursor, duration, output = None, 0, []
    for shot in shots:
        gap = 0 if cursor is None else max(0, shot["start"]-cursor)
        if gap:
            output.append({"kind": "gap", "duration": 2, "elapsed_seconds": gap, "start": cursor, "end": shot["start"], "offset": duration})
            duration += 2
        shot = shot | {"kind": "video", "offset": duration, "duration": shot["end"]-shot["start"]}
        output.append(shot)
        duration += shot["duration"]
        cursor = shot["end"]
    return {"shots": output, "duration": duration, "missing_coverage": missing[:256], "missing_coverage_truncated": len(missing)>256, "directed": directed, "split_screen": split_screen, "audio": "muted", "evidence_fingerprint": evidence_fingerprint(incidents)}
