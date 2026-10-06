import React, { useEffect, useRef, useState } from "react";
import { fetch, appUrl } from "../shared/api.js";
import { NativeRecordingVideo } from "../shared/NativeRecordingVideo.jsx";
import { storyShotAt, storyCropAt, nextStoryPosition } from "../storyReplay.mjs";

function segmentUrl(camera, at, mobile = false) {
  return appUrl(`/api/cameras/${encodeURIComponent(camera)}/recordings/segment.mp4?${new URLSearchParams({ epoch: at, source: "main", ...(mobile ? { mobile: "true" } : {}) })}`);
}

function StoryView({ view, shot, epoch, playing, fullFrame, primary, onEpoch, onEnded, onError }) {
  const videoRef = useRef(null), stageRef = useRef(null);
  const [loadedBucket, setLoadedBucket] = useState(null);
  const [rows, setRows] = useState([]), [dimensions, setDimensions] = useState(null);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const [error, setError] = useState(""), [mobile, setMobile] = useState(false), [retry, setRetry] = useState(0);
  const bucket = Math.floor(epoch / 900) * 900;
  useEffect(() => {
    const controller = new AbortController();
    setError(""); setRows([]); setLoadedBucket(null);
    void fetch(`/api/cameras/${encodeURIComponent(view.camera_id)}/recordings/window?${new URLSearchParams({ start_epoch: bucket, end_epoch: bucket + 900, source: "main" })}`, { signal: controller.signal })
      .then(async (response) => { const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Recording unavailable"); return data; })
      .then((data) => { setRows(data.recordings || []); setLoadedBucket(bucket); })
      .catch((failure) => { if (failure.name !== "AbortError") { setError(String(failure.message)); onError(); } });
    return () => controller.abort();
  }, [view.camera_id, bucket, retry]);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setSize({ width: entry.contentRect.width, height: entry.contentRect.height }));
    observer.observe(stageRef.current); return () => observer.disconnect();
  }, []);
  const row = rows.find((r) => Number(r.start_epoch) <= epoch && epoch < Number(r.end_epoch));
  useEffect(() => {
    if (loadedBucket === bucket && !row) { setError("This recording is no longer available. Refresh the Story Replay."); onError(); }
  }, [loadedBucket, bucket, row]);
  const nextRow = row && rows.find((r) => Number(r.start_epoch) >= Number(row.end_epoch) - .01 && r !== row);
  const src = row ? segmentUrl(view.camera_id, row.start_epoch, mobile) : "";
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !row || video.readyState < 1) return;
    const target = Math.max(0, epoch - Number(row.start_epoch));
    if (!video.seeking && Math.abs(video.currentTime - target) > (primary && playing ? 1 : .3)) video.currentTime = target;
    if (playing) video.play().catch(() => { setError("Playback needs a tap. Pause and press Play again."); onError(); });
    else video.pause();
  }, [epoch, playing, row, dimensions, primary]);
  const crop = fullFrame ? { x: .5, y: .5, size: 1 } : storyCropAt(view.crop, epoch - shot.start + (view.crop_offset || 0));
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !dimensions) return;
    const fit = Math.min(size.width / dimensions.width, size.height / dimensions.height);
    const width = dimensions.width * fit / crop.size, height = dimensions.height * fit / crop.size;
    for (const video of stage.querySelectorAll("video")) Object.assign(video.style, { width: `${width}px`, height: `${height}px`, left: `${size.width / 2 - crop.x * width}px`, top: `${size.height / 2 - crop.y * height}px` });
  }, [dimensions, size, crop.x, crop.y, crop.size]);
  return <div className="story-view" ref={stageRef}>
    <NativeRecordingVideo ref={videoRef} src={src} nextSrc={nextRow ? segmentUrl(view.camera_id, nextRow.start_epoch, mobile) : ""} autoPlay={playing} muted
      onLoadedMetadata={(event) => {
        const video = event.currentTarget;
        if (row) video.currentTime = Math.max(0, epoch - Number(row.start_epoch));
        setDimensions({ width: video.videoWidth, height: video.videoHeight });
      }}
      onTimeUpdate={(event) => { if (primary && playing && row && !event.currentTarget.seeking) onEpoch(Number(row.start_epoch) + event.currentTarget.currentTime); }}
      onEnded={() => { if (primary && row) { if (Number(row.end_epoch) >= shot.end - .1) onEnded(); else onEpoch(Number(row.end_epoch)); } }}
      onError={() => { if (!mobile) setMobile(true); else { setError("This recording could not be decoded."); onError(); } }} />
    <div className="story-view-label">{view.camera_name || view.camera_id}<br />{new Date(epoch * 1000).toISOString().replace("T", " ").slice(0, 19)} UTC</div>
    {!row && !error ? <div className="story-video-status">Loading recording…</div> : null}
    {error ? <div className="story-video-status" role="alert">{error}<button onClick={() => { setError(""); setRetry((n) => n + 1); }}>Retry footage</button></div> : null}
  </div>;
}

export function StoryReplay({ plan }) {
  const [position, setPosition] = useState(0), [playing, setPlaying] = useState(false), [fullFrame, setFullFrame] = useState(false);
  const shot = storyShotAt(plan, position);
  useEffect(() => { setPosition(0); setPlaying(false); }, [plan]);
  useEffect(() => {
    if (!playing || shot?.kind !== "gap") return undefined;
    const timer = window.setInterval(() => setPosition((p) => Math.min(plan.duration, p + .1)), 100);
    return () => window.clearInterval(timer);
  }, [playing, shot, plan.duration]);
  useEffect(() => { if (position >= plan.duration) setPlaying(false); }, [position, plan.duration]);
  function next() { setPosition(nextStoryPosition(plan, shot)); }
  return <section className="story-replay" aria-label="Story Replay">
    <div className={`story-replay-stage ${shot?.views?.length === 2 ? "split" : ""}`}>
      {shot?.kind === "video" ? shot.views.map((view, index) => <StoryView key={`${shot.offset}:${view.camera_id}`} view={view} shot={shot} epoch={Math.min(shot.end - .001, shot.start + position - shot.offset)} playing={playing} fullFrame={fullFrame} primary={index === 0} onEpoch={(at) => { if (at >= shot.end - .05) next(); else setPosition(Math.max(shot.offset, shot.offset + at - shot.start)); }} onEnded={next} onError={() => setPlaying(false)} />)
        : <div className="story-gap">{shot?.kind === "gap" ? <><strong>{shot.elapsed_seconds >= 0 ? `${shot.elapsed_seconds.toFixed(1)} seconds elapsed` : `Source time moves back ${Math.abs(shot.elapsed_seconds).toFixed(1)} seconds`}</strong><span>No selected footage shown</span></> : <strong>{plan.shots.length ? "Replay complete" : "No retained main recordings"}</strong>}</div>}
    </div>
    <div className="story-controls"><button disabled={!plan.shots.length} onClick={() => { if (position >= plan.duration) setPosition(0); setPlaying(!playing); }}>{playing ? "Pause" : "Play Story"}</button><input aria-label="Story Replay position" type="range" min="0" max={plan.duration || 1} step="0.1" value={position} onChange={(e) => setPosition(Number(e.target.value))} /><span>{Math.floor(position)} / {Math.ceil(plan.duration)}s</span><label><input type="checkbox" checked={fullFrame} onChange={(e) => setFullFrame(e.target.checked)} />Full frame</label></div>
    {plan.missing_coverage?.length || plan.missing_incidents?.length ? <p className="story-warning">Some source footage or incidents are unavailable. Review coverage details below.</p> : null}
    <details><summary>Replay chapters and coverage</summary>{plan.shots.map((s) => <button key={s.offset} onClick={() => { setPlaying(false); setPosition(s.offset); }}>{Math.floor(s.offset)}s · {s.kind === "gap" ? `${s.elapsed_seconds.toFixed(1)}s elapsed` : s.views.map((v) => v.camera_name || v.camera_id).join(" + ")}</button>)}{plan.missing_coverage.map((g, n) => <p key={n}>Missing footage: {g.camera_id}, {new Date(g.start * 1000).toISOString()} to {new Date(g.end * 1000).toISOString()}</p>)}</details>
  </section>;
}
