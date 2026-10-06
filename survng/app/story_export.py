"""Render a frozen replay plan through the existing bounded export worker."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .recording_media import concatenated_clip_timing


def _piecewise(points, key, offset):
    if not points:
        return "1" if key == "size" else "0.5"
    t = f"(on/25+{offset:.6f})"
    expression = f"{points[-1][key]:.6f}"
    for left, right in reversed(list(zip(points, points[1:]))):
        u = f"clip(({t}-{left['at']:.6f})/{max(.001,right['at']-left['at']):.6f},0,1)"
        smooth = f"({u}*{u}*(3-2*{u}))"
        value = f"({left[key]:.6f}+({right[key]-left[key]:.6f})*{smooth})"
        expression = f"if(lte({t},{right['at']:.6f}),{value},{expression})"
    default = "1" if key == "size" else "0.5"
    return f"if(lt({t},{points[0]['at']:.6f})+gt({t},{points[-1]['at']:.6f}),{default},{expression})"


def render_storyline(owner, job, work, cancel):
    plan = job["options"]["replay_plan"]
    shots = plan.get("shots", [])
    if not shots or len(shots)>512 or sum(s["duration"] for s in shots)>4200:
        raise RuntimeError("Story Replay plan exceeds rendering limits")
    outputs, gaps = [], list(plan.get("missing_coverage", []))
    for index, shot in enumerate(shots):
        if cancel.is_set() or owner._stop.is_set():
            raise InterruptedError
        output = work/f"story-{index:04d}.mp4"
        owner.store.update(str(job["id"]), phase=f"Rendering Story Replay {index+1} of {len(shots)}", progress=5+80*index/len(shots))
        command = [owner._ffmpeg_path(), "-hide_banner", "-loglevel", "warning"]
        filters = []
        if shot["kind"] == "gap":
            gap_label = f"{shot['elapsed_seconds']:.1f} seconds elapsed" if shot["elapsed_seconds"] >= 0 else f"Source time moves back {abs(shot['elapsed_seconds']):.1f} seconds"
            text = f"{gap_label}\nNo selected footage shown\n{datetime.fromtimestamp(shot['start'], timezone.utc).isoformat()}"
            caption = work/f"caption-{index}.txt"
            caption.write_text(text, encoding="utf-8")
            command += ["-f", "lavfi", "-i", "color=c=0x101824:s=1280x720:r=25"]
            filters = [f"[0:v]drawtext=textfile='{caption}':expansion=none:fontcolor=white:fontsize=28:x=(w-tw)/2:y=(h-th)/2[v]"]
            gaps.append({"start": shot["start"], "end": shot["end"], "elapsed_seconds": shot["elapsed_seconds"], "reason": "No selected footage shown"})
        else:
            views = shot["views"]
            if not 1 <= len(views) <= 2:
                raise RuntimeError("Invalid Story Replay view count")
            pane_width = 1280 if len(views)==1 else 640
            for view_index, view in enumerate(views):
                rows = owner._recorder().recording_rows_between(view["camera_id"], shot["start"], shot["end"], "main", discover_missing=False)
                rows = [r for r in rows if Path(str(r.get("path", ""))).is_file()]
                groups, coverage_gaps = owner._continuous_groups(rows, shot["start"], shot["end"])
                if not groups or coverage_gaps or groups[0][0]["start_epoch"]>shot["start"]+.05 or groups[-1][-1]["end_epoch"]<shot["end"]-.05:
                    raise RuntimeError("Story Replay recording coverage changed; refresh the replay and retry")
                # Stream fingerprint changes are safe only after normalizing each span.
                # A fresh plan can cut at that boundary rather than joining incompatible streams.
                if len(groups)!=1:
                    raise RuntimeError("Recording format changed inside a replay shot; use separate replay intervals")
                rows = groups[0]
                owner._recorder().lease_recordings_for_playback(rows, ttl_seconds=max(900, shot["duration"]*4+300))
                concat = owner._write_concat(rows, work/f"story-{index}-{view_index}.ffconcat")
                local_start, duration = concatenated_clip_timing(rows, shot["start"], shot["end"])
                command += ["-f", "concat", "-safe", "0", "-i", str(concat)]
                profile = owner._probe_source(Path(str(rows[0]["path"])))
                width, height = owner._target_dimensions(profile, 540)
                crop = view.get("crop", [])
                # zoompan's on counter is output-frame time, independent of input FPS.
                size = _piecewise(crop, "size", view.get("crop_offset", 0))
                x = _piecewise(crop, "x", view.get("crop_offset", 0))
                y = _piecewise(crop, "y", view.get("crop_offset", 0))
                size = f"(1+(({size})-1)*clip((on/25+{view.get('crop_offset',0):.6f}-1)/2,0,1))"
                caption = work/f"caption-{index}-{view_index}.txt"
                caption.write_text(f"{view.get('camera_name',view['camera_id'])}\nUTC {datetime.fromtimestamp(shot['start'],timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(shot['end'],timezone.utc).strftime('%H:%M:%S')}", encoding="utf-8")
                label = f"pane{view_index}"
                filters.append(f"[{view_index}:v]trim=start={local_start:.6f}:duration={duration:.6f},setpts=PTS-STARTPTS,fps=25,zoompan=z='1/({size})':x='max(0,min(iw-iw/zoom,iw*({x})-iw/zoom/2))':y='max(0,min(ih-ih/zoom,ih*({y})-ih/zoom/2))':d=1:s={width}x{height}:fps=25,scale={pane_width}:720:force_original_aspect_ratio=decrease,pad={pane_width}:720:(ow-iw)/2:(oh-ih)/2,setsar=1,drawtext=textfile='{caption}':expansion=none:fontcolor=white:fontsize=16:box=1:boxcolor=black@0.6:x=8:y=8[{label}]")
            if len(views)==2:
                filters.append("[pane0][pane1]hstack=inputs=2[v]")
            else:
                filters.append("[pane0]null[v]")
        command += ["-filter_complex_threads", "1", "-filter_complex", ";".join(filters), "-map", "[v]", "-t", f"{shot['duration']:.6f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(output)]
        owner._run_process(command, cancel, timeout=max(120, min(14400, shot["duration"]*4)), process_name="survng-story-export")
        if not owner._valid_video_output(output):
            raise RuntimeError("Story Replay produced no playable video")
        outputs.append(output)
    concat = owner._write_concat([{"path": str(p)} for p in outputs], work/"story-final.ffconcat")
    output = work/"storyline.mp4"
    owner._run_process([owner._ffmpeg_path(), "-hide_banner", "-loglevel", "warning", "-f", "concat", "-safe", "0", "-i", str(concat), "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart", "-y", str(output)], cancel, timeout=180, process_name="survng-story-export")
    if not owner._valid_video_output(output):
        raise RuntimeError("Story Replay could not join rendered shots")
    return output, gaps
