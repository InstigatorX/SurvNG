import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Camera, ChevronLeft, ChevronRight, ImageOff, Play, RefreshCw, X } from "lucide-react";
import { eventCoverEpoch, incidentGalleryFrames, incidentEvidenceFrames } from "../incidentNavigation.mjs";
import { appUrl, fetch, recordingsHref } from "../shared/api.js";
import { formatDateTime, formatTimeOnly } from "../shared/format.js";
import { recordingPreviewUrl } from "../shared/mediaUrls.js";
import { observationBoxStyle, canonicalIncidentHref, incidentEpisodeClips } from "../incidentScene.mjs";
import {
  cardIdentities, epochOf, estimatedBox, findObservation, followSegments, formatSpan, frameBoxes, incidentStoryTitle, nearestObservation,
  objectBestObservation, objectName, observationSnapshotUrl, reviewKeyAction, reviewPosition, sameObservationFrame, storyObjects, storyline,
} from "../incidentStory.mjs";
import { IncidentRecordingPlayer } from "./IncidentRecordingPlayer.jsx";
import { IncidentScenePanel } from "./IncidentScenePanel.jsx";
import { IncidentSpeechPanel } from "./IncidentSpeech.jsx";
import { IncidentStoryline } from "./IncidentStoryline.jsx";
import { CorrectPanel, ObjectCards, PathPanel, WhyPanel } from "./IncidentStoryObjects.jsx";
import { IncidentPeople } from "../people/IncidentPeople.jsx";
import { useIncidentStill } from "./useIncidentStill.js";
import "./incident-detail.css";
import "./incident-story.css";

const VERDICTS = [
  { verdict: "reviewed", label: "Reviewed", key: "E" },
  { verdict: "keep", label: "Keep for later", key: "F" },
  { verdict: "nothing_here", label: "Nothing here", key: "X" },
];

const NEARBY_SECONDS = 2;
const SAME_MOMENT_SECONDS = 0.1;

function eventThumbnail(eventId, revision) {
  return eventId ? `/api/events/${eventId}/thumbnail.jpg?width=1280&quality=85&revision=${encodeURIComponent(revision)}` : "";
}

function recordingCandidate(cameraId, epoch, source) {
  return { src: recordingPreviewUrl(encodeURIComponent(cameraId), epoch, source, { width: 1280, exact: true }), epoch, source };
}

function nearbyStill(observations, cameraId, epoch) {
  const candidates = observations.filter((observation) => observationSnapshotUrl(observation)
    && Math.abs(epochOf(observation.captured_at) - epoch) <= NEARBY_SECONDS);
  return nearestObservation({ observations: candidates }, epoch, cameraId);
}

function savedCandidate(observation) {
  return { src: observationSnapshotUrl(observation), epoch: epochOf(observation?.captured_at), observation };
}

function stillCandidates({ cameraId, epoch, saved, nearby }) {
  const frames = [];
  if (saved.src) frames.push(saved);
  if (cameraId && Number.isFinite(epoch)) {
    frames.push(recordingCandidate(cameraId, epoch, "main"), recordingCandidate(cameraId, epoch, "live"));
  }
  if (nearby.src && nearby.src !== saved.src) frames.push(nearby);
  return frames;
}

function EvidenceImage({ frame, onClick }) {
  const [failed, setFailed] = useState(null);
  const candidate = failed === frame.src ? frame.fallback : frame;
  const label = candidate?.label || frame.label;
  const src = candidate?.src;
  const content = !src || failed === frame.fallback?.src
    ? <span className="incident-detail-image-empty"><ImageOff size={28} />Image unavailable</span>
    : <img key={src} loading={onClick ? "lazy" : "eager"} src={appUrl(src)} alt={label} onError={() => setFailed(src)} />;
  return onClick
    ? <button className="incident-detail-frame" onClick={onClick} aria-label={label}>{content}<span>{label}</span></button>
    : <><div className="incident-detail-hero">{content}</div><p className="incident-detail-hint">{label}</p></>;
}

function EvidenceViewer({ frames, selected, onClose }) {
  const dialog = useRef(null);
  const [index, setIndex] = useState(selected);
  useEffect(() => { dialog.current.showModal(); }, []);
  const frame = frames[index];
  return <dialog ref={dialog} className="incident-evidence-viewer" aria-label="Evidence image viewer" onCancel={onClose} onClose={onClose}>
    <header><span>{index + 1} / {frames.length}</span><button autoFocus onClick={onClose} aria-label="Close evidence"><X size={22} /></button></header>
    <EvidenceImage key={frame.src} frame={frame} />
    <nav aria-label="Evidence navigation"><button disabled={index === 0} onClick={() => setIndex(index - 1)}>Previous</button><button disabled={index === frames.length - 1} onClick={() => setIndex(index + 1)}>Next</button></nav>
  </dialog>;
}

// A still frame with every object's box from the same capture, colored by object.
function StillFrame({ still, anchor, boxes, selectedId, legacyBoxes, label, onCover }) {
  const width = Number(still.width || anchor?.detection_frame_width || legacyBoxes?.[0]?.detection_frame_width) || 16;
  const height = Number(still.height || anchor?.detection_frame_height || legacyBoxes?.[0]?.detection_frame_height) || 9;
  if (still.status === "loading") return <span role="status" className="incident-detail-image-empty story-frame-empty">Loading recording frame…</span>;
  if (still.status === "unavailable") return <div className="incident-detail-image-empty story-frame-empty"><ImageOff size={28} />Image unavailable
    {onCover ? <button type="button" onClick={onCover}>Back to best frame</button> : null}</div>;
  return <div className="story-frame-image" style={{ aspectRatio: `${width} / ${height}`, "--frame-w": width, "--frame-h": height }}>
    <img key={still.src} src={appUrl(still.src)} alt={label} onError={still.onError} />
    {boxes.map(({ object, observation }) => {
      const style = observationBoxStyle(observation);
      if (!style) return null;
      const dimmed = selectedId != null && String(object.id) !== String(selectedId);
      return <span key={observation.id} className={`story-box ${object.kind}${dimmed ? " dimmed" : ""}${observation.estimated ? " estimated" : ""}`} style={style}>
        <span>{objectName(object)}{object.kind === "uncertain" ? "? · uncertain" : ""}{observation.estimated ? " · estimated" : ""}</span>
      </span>;
    })}
    {(legacyBoxes || []).map((item, index) => {
      const style = observationBoxStyle(item);
      return style ? <span key={`legacy-${index}`} className={`story-box ${item.label === "person" ? "person" : "other"}`} style={style}><span>{objectName(item)}</span></span> : null;
    })}
  </div>;
}

function localDay(epoch, timeZone) {
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone, year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(new Date(epoch * 1000));
  const get = (type) => parts.find((part) => part.type === type)?.value;
  return `${get("year")}-${get("month")}-${get("day")}`;
}

function dayLabel(epoch, timeZone) {
  return new Intl.DateTimeFormat("en-US", { timeZone, weekday: "short", month: "short", day: "numeric" }).format(new Date(epoch * 1000));
}

export function safeDecode(value) {
  try { return decodeURIComponent(value); } catch { return value; }
}

function routeState() {
  const params = new URLSearchParams(window.location.search);
  const match = window.location.pathname.match(/\/incidents\/(incident-[^/]+)\/?$/);
  return { id: match ? safeDecode(match[1]) : null, review: params.get("review") === "1" };
}

export function IncidentDetailPage({ incidentId, timeZone, canCorrectIncident = false }) {
  const [faceNames, setFaceNames] = useState([]);
  // Scene object id -> names from face capture (faces bound to that tracked person).
  const [faceObjects, setFaceObjects] = useState({});
  const [currentId, setCurrentId] = useState(incidentId);
  const [reviewRequested, setReview] = useState(() => routeState().review);
  // Review mode is for those who can record verdicts.
  const review = reviewRequested && canCorrectIncident;
  // Loaded data remembers which incident it belongs to, so a render right
  // after navigation never pairs the new incident with the previous details.
  const [loaded, setLoaded] = useState(null);
  const data = loaded?.id === currentId ? loaded.payload : null;
  // A correction response merges only into the incident it was made on.
  const updateIncident = (detail, forId) => setLoaded((current) => (current && current.id === forId
    ? { ...current, payload: { ...current.payload, incident: { ...current.payload.incident, ...detail } } } : current));
  const currentIdRef = useRef(incidentId);
  const titleRef = useRef(null);
  const topRef = useRef(null);
  const frameElementRef = useRef(null);
  const sightingsHeadingRef = useRef(null);
  const focusTitleRef = useRef(false);
  const [error, setError] = useState("");
  const [missing, setMissing] = useState(false);
  const [retry, setRetry] = useState(0);
  const [cameraNames, setCameraNames] = useState(() => new Map());
  const [selectedId, setSelectedId] = useState(null);
  const [correcting, setCorrecting] = useState(false);
  const [frameRef, setFrameRef] = useState(null); // camera and moment pinned by a bar click
  const [cursor, setCursor] = useState(null);
  const [playback, setPlayback] = useState(null);
  const [notice, setNotice] = useState(null);
  const [queue, setQueue] = useState(null);
  const [queueRevision, setQueueRevision] = useState(0);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [reviewError, setReviewError] = useState("");
  const [selectedFrame, setSelectedFrame] = useState(null);
  const zone = timeZone || Intl.DateTimeFormat().resolvedOptions().timeZone;

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/cameras", { signal: controller.signal }).then((response) => response.ok ? response.json() : []).then((cameras) => {
      setCameraNames(new Map((Array.isArray(cameras) ? cameras : []).map((camera) => [camera.id, camera.name || camera.id])));
    }).catch(() => {});
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const restore = () => {
      const state = routeState();
      if (state.id) setCurrentId(state.id);
      setReview(state.review);
    };
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);

  useEffect(() => {
    setMissing(false); setError("");
    currentIdRef.current = currentId;
    setSelectedId(null); setCorrecting(false); setFrameRef(null); setCursor(null); setPlayback(null); setNotice(null); setReviewError(""); setSelectedFrame(null);
    setFaceNames([]);
    setFaceObjects({}); // the next incident's faces are looked up afresh
  }, [currentId]);

  useEffect(() => {
    let alive = true;
    let busy = false;
    const controller = new AbortController();
    async function load() {
      if (busy || document.hidden) return;
      busy = true;
      try {
        const response = await fetch(`/api/incidents/notification/${encodeURIComponent(currentId)}`, { signal: controller.signal });
        if (response.status === 404) {
          if (alive) { setMissing(true); setLoaded(null); setError(""); }
          return;
        }
        if (!response.ok) throw new Error("Incident request failed");
        const result = await response.json();
        if (alive) {
          // A poll that started before a correction saved must not undo it.
          setLoaded((current) => (current?.id === currentId && Number(current.payload?.incident?.revision) > Number(result?.incident?.revision)
            ? current : { id: currentId, payload: result }));
          setError(""); setMissing(false);
        }
      } catch (failure) {
        if (alive && failure.name !== "AbortError") setError("Could not refresh this incident. Check your connection and try again.");
      } finally { busy = false; }
    }
    void load();
    const timer = window.setInterval(load, 5000);
    document.addEventListener("visibilitychange", load);
    window.addEventListener("online", load);
    return () => {
      alive = false;
      controller.abort();
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", load);
      window.removeEventListener("online", load);
    };
  }, [currentId, retry]);

  const incident = data?.incident;
  const notification = data?.notification;
  const cameraName = useCallback((id) => cameraNames.get(id) || (id === incident?.camera_id ? data?.camera_name : null) || id || "Camera",
    [cameraNames, data?.camera_name, incident?.camera_id]);
  const objects = useMemo(() => storyObjects(incident), [incident]);
  const observations = useMemo(() => objects.flatMap((object) => object.observations || []), [objects]);
  const line = useMemo(() => storyline(incident, objects), [incident, objects]);
  const selected = objects.find((object) => String(object.id) === String(selectedId)) || null;
  // A correction or a refresh can remove the selected object; then nothing is selected.
  const focusId = selected ? selectedId : null;
  const startEpoch = epochOf(incident?.start_at) || Number(incident?.start_epoch) || line.start;
  const endEpoch = epochOf(incident?.end_at) || Number(incident?.last_epoch) || line.end;
  const day = incident && Number.isFinite(startEpoch) && startEpoch > 0 ? localDay(startEpoch, zone) : "";

  useEffect(() => {
    if (!day) return undefined;
    const controller = new AbortController();
    fetch(`/api/incidents/review-queue?${new URLSearchParams({ day, time_zone: zone })}`, { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null).then((result) => setQueue(result)).catch(() => {});
    return () => controller.abort();
  }, [day, zone, queueRevision]);
  const position = reviewPosition(queue, incident?.incident_id || currentId);

  // Optional provider-worded title (Admin → AI Provider, off by default). The
  // built-in title shows immediately and stays when this is off or fails.
  const [aiTitle, setAiTitle] = useState(null);
  const titleId = incident?.incident_id;
  useEffect(() => setAiTitle(null), [titleId]);
  // A growing incident keeps its built-in title, which follows every change;
  // the AI wording is asked for once the incident is no longer ongoing.
  const titleSettled = Boolean(incident) && !["active", "new", "updated"].includes(notification?.state || incident?.state);
  useEffect(() => {
    if (!titleId || !titleSettled) return undefined;
    const controller = new AbortController();
    let timer = 0, attempts = 0;
    const load = () => fetch(`/api/incidents/${encodeURIComponent(titleId)}/title`, { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((result) => {
        if (!result) return;
        // Keep a shown wording while a newer one is pending or busy.
        if (["pending", "busy"].includes(result.status)) {
          if (++attempts < 6) timer = window.setTimeout(load, 5000);
          return;
        }
        setAiTitle({ ...result, id: titleId });
        // A wording of an older revision is not shown; ask again once the
        // server may reword it (at most once a minute).
        if (result.status === "ready" && result.current === false && ++attempts < 6) {
          timer = window.setTimeout(load, ["pending", "busy"].includes(result.refresh) ? 5000 : 65000);
        }
      })
      .catch((failure) => { if (failure.name !== "AbortError") setAiTitle({ id: titleId, status: "failed", message: "AI title unavailable" }); });
    void load();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [titleId, incident?.revision, titleSettled]);

  const navigate = useCallback((id, nextReview = review) => {
    if (!id) return;
    const url = appUrl(`/incidents/${encodeURIComponent(id)}${nextReview ? "?review=1" : ""}`);
    window.history.pushState(null, "", url);
    focusTitleRef.current = true;
    setReview(nextReview);
    setCurrentId(id);
  }, [review]);

  function toggleReview() {
    const next = !review;
    window.history.replaceState(null, "", appUrl(`/incidents/${encodeURIComponent(currentId)}${next ? "?review=1" : ""}`));
    setReview(next);
    setReviewError("");
  }

  async function decide(verdict) {
    if (!incident?.incident_id || reviewBusy) return;
    const requestedFor = currentId;
    setReviewBusy(true); setReviewError("");
    try {
      const response = await fetch(`/api/incidents/${encodeURIComponent(incident.incident_id)}/review`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ verdict }),
      });
      if (!response.ok) throw new Error(response.status === 403 ? "Only an administrator can record a verdict." : "Could not save the verdict. Try again.");
      setQueueRevision((value) => value + 1);
      const updated = queue && { ...queue, items: queue.items.map((item) => item.incident_id === incident.incident_id ? { ...item, verdict } : item) };
      if (updated) setQueue(updated);
      // Only act if the person is still reviewing the incident they decided.
      if (currentIdRef.current !== requestedFor || !reviewRef.current) return;
      if (!updated) { setReviewError("Verdict saved. The review queue is unavailable; use Next to continue."); return; }
      const target = reviewPosition(updated, incident.incident_id).nextUndecided;
      if (target) navigate(target, true);
      else setReviewError("Every incident on this day has a verdict.");
    } catch (failure) {
      if (currentIdRef.current === requestedFor) setReviewError(failure.message);
    } finally { setReviewBusy(false); }
  }

  const reviewRef = useRef(review);
  reviewRef.current = review;
  const keyHandler = useRef(null);
  keyHandler.current = (event) => {
    if (!review) return;
    const action = reviewKeyAction(event);
    if (!action) return;
    event.preventDefault();
    if (reviewBusy) return; // a verdict in flight decides where to go next
    if (action === "next") navigate(position.next, true);
    else if (action === "previous") navigate(position.previous, true);
    else if (canCorrectIncident) void decide(action);
  };
  useEffect(() => {
    const listener = (event) => keyHandler.current?.(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(null), 4000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const cover = incident?.snapshot_observation_id ? findObservation(objects, incident.snapshot_observation_id)?.observation : null;
  const coverEvent = incident?.events?.find((event) => String(event.id) === String(incident.representative_event_id));
  const coverEpoch = epochOf(cover?.captured_at || incident?.snapshot_captured_at) || eventCoverEpoch(coverEvent || incident) || startEpoch;
  const revision = notification?.revision || incident?.end_at || 0;
  const pin = selected ? frameRef : null;
  const pinned = pin ? nearestObservation(selected, pin.epoch, pin.cameraId) : null;
  const selectedAnchor = selected && (cover && frameBoxes([selected], cover).length ? cover : objectBestObservation(selected));
  const targetCamera = pin?.cameraId || selectedAnchor?.camera_id || selected?.intervals[0]?.camera_id || cover?.camera_id || coverEvent?.camera_id || incident?.camera_id;
  const targetEpoch = pin?.epoch ?? (epochOf(selectedAnchor?.captured_at) || selected?.intervals[0]?.start || coverEpoch);
  const coverStill = cover ? savedCandidate(cover) : {
    src: incident?.snapshot_url || eventThumbnail(incident?.representative_event_id, revision),
    epoch: coverEpoch,
    legacy: true,
  };
  const isCover = !selected;
  const targetObservation = pin
    ? (Math.abs(epochOf(pinned?.captured_at) - targetEpoch) < SAME_MOMENT_SECONDS ? pinned : null)
    : selectedAnchor;
  const sharedStill = targetObservation && !observationSnapshotUrl(targetObservation)
    ? observations.find((item) => observationSnapshotUrl(item) && sameObservationFrame(item, targetObservation)) : null;
  const saved = isCover ? coverStill : savedCandidate(sharedStill || targetObservation);
  const nearby = savedCandidate(nearbyStill(selected?.observations || [], targetCamera, targetEpoch));
  const still = useIncidentStill(playback ? [] : stillCandidates({ cameraId: targetCamera, epoch: targetEpoch, saved, nearby }));
  const anchor = still.observation;
  const shownBoxes = frameBoxes(objects, anchor);
  // A recording frame between analysed frames has no box of its own. Keep the
  // followed object marked with an estimate, shown as one.
  const estimate = selected && still.status === "ready" && !still.legacy
    && !shownBoxes.some(({ object }) => String(object.id) === String(selected.id))
    ? estimatedBox(selected, targetCamera, still.epoch ?? targetEpoch) : null;
  const boxes = estimate ? [...shownBoxes, { object: selected, observation: estimate }] : shownBoxes;
  const legacyBoxes = still.legacy ? (incident?.objects || []).filter((item) => item.box) : null;
  const frameCamera = playback ? playback.segments[playback.index]?.cameraId : targetCamera;
  const shownEpoch = playback && Number.isFinite(cursor) ? cursor : (still.epoch ?? targetEpoch);
  let mode = selected ? `Following ${objectName(selected)}` : "Best frame";
  if (still.status === "ready" && still.source) {
    mode = still.actual == null ? "Recording frame · approximate time"
      : Math.abs(still.actual - targetEpoch) >= SAME_MOMENT_SECONDS ? "Nearby recording frame" : "Recording frame";
  } else if (still.status === "ready" && Math.abs(still.epoch - targetEpoch) >= SAME_MOMENT_SECONDS) {
    mode = "Nearby saved frame";
  } else if (still.status === "ready" && selected && selected.observations?.length === 1) {
    mode = "Only supporting image";
  }
  if (selected && still.status === "ready" && still.source) mode = `Following ${objectName(selected)} · ${mode.charAt(0).toLowerCase()}${mode.slice(1)}`;

  function segmentsFor(object) {
    if (object) return followSegments(object, { start: startEpoch, end: endEpoch });
    return incidentEpisodeClips(incident).filter((episode) => episode.clip)
      .map((episode) => ({ cameraId: episode.camera_id, episodeId: episode.episode_id, start: episode.clip.start_epoch, end: episode.clip.last_epoch }));
  }
  const canPlay = segmentsFor(selected).length > 0;

  // Play from the moment pinned on the storyline when there is one.
  function startFollowing() {
    const segments = segmentsFor(selected);
    const pinnedAt = pin?.epoch ?? NaN;
    const index = Number.isFinite(pinnedAt)
      ? segments.findIndex((item) => item.cameraId === pin.cameraId && pinnedAt >= item.start && pinnedAt <= item.end) : -1;
    startPlayback(segments, Math.max(0, index), index >= 0 ? pinnedAt : null, null);
  }

  function startPlayback(segments, index = 0, initialEpoch = null, from = null) {
    if (!segments.length) return;
    setFrameRef(null);
    setCursor(initialEpoch ?? segments[index].start);
    setPlayback({ segments, index, initialEpoch, key: Date.now() });
    if (from && from !== segments[index].cameraId) setNotice({ text: `Switched from ${cameraName(from)}`, key: Date.now() });
  }

  function stopPlayback() {
    // The still frame shows its own moment again, not the last playhead.
    setPlayback(null);
    setCursor(null);
  }

  function playbackEnded(skipped = false) {
    const current = playback;
    if (!current) return;
    const next = current.index + 1;
    if (next >= current.segments.length) { stopPlayback(); return; }
    const from = current.segments[current.index].cameraId;
    const to = current.segments[next].cameraId;
    if (skipped) setNotice({ text: `No recording on ${cameraName(from)}; ${from !== to ? `switched to ${cameraName(to)}` : "moving on"}`, key: Date.now() });
    else if (from !== to) setNotice({ text: `Switched from ${cameraName(from)}`, key: Date.now() });
    setPlayback({ ...current, index: next, initialEpoch: Math.max(current.segments[next].start, current.segments[current.index].end - 2), key: Date.now() });
  }

  function selectObject(id) {
    setSelectedId(id);
    setCorrecting(false);
    setFrameRef(null);
    setPlayback(null);
    setCursor(null);
    setNotice(null);
  }

  function jumpToBar(bar, epoch) {
    const object = objects.find((item) => String(item.id) === String(bar.objectId));
    const from = frameCamera;
    setSelectedId(bar.objectId);
    setCorrecting(false);
    if (playback) {
      const segments = segmentsFor(object);
      let index = segments.findIndex((segment) => segment.cameraId === bar.camera_id && epoch >= segment.start - 1 && epoch <= segment.end + 1);
      if (index < 0) {
        const distance = (segment) => Math.min(Math.abs(segment.start - epoch), Math.abs(segment.end - epoch));
        index = segments.reduce((best, segment, candidate) => (segment.cameraId === bar.camera_id && (best < 0 || distance(segment) < distance(segments[best])) ? candidate : best), -1);
      }
      if (index >= 0) {
        startPlayback(segments, index, epoch, from);
        return;
      }
      stopPlayback();
    }
    setFrameRef({ cameraId: bar.camera_id, epoch });
    if (from && from !== bar.camera_id) setNotice({ text: `Switched from ${cameraName(from)}`, key: Date.now() });
  }

  const evidenceFrames = useMemo(() => {
    if (!incident) return [];
    const seen = new Set();
    const events = [...(incident.events || [])].sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
    const existing = events.flatMap((event) => incidentEvidenceFrames(event).map((frame) => {
      const cameraId = event.camera_id || incident.camera_id;
      const retained = frame.kind !== "snapshot" && nearbyStill(observations, cameraId, frame.epoch);
      const eventEpoch = eventCoverEpoch(event);
      const thumbnail = eventThumbnail(event.id, revision);
      const timeLabel = (epoch) => formatDateTime(new Date(epoch * 1000).toISOString(), zone);
      const src = frame.kind === "snapshot"
        ? thumbnail : recordingCandidate(cameraId, frame.epoch, "main").src;
      const fallback = retained
        ? { src: observationSnapshotUrl(retained), label: `Saved frame · ${timeLabel(epochOf(retained.captured_at))}` }
        : frame.kind === "snapshot" ? null
          : Math.abs(eventEpoch - frame.epoch) <= NEARBY_SECONDS
            ? { src: thumbnail, label: `Saved frame · ${timeLabel(eventEpoch)}` }
            : { src: recordingCandidate(cameraId, frame.epoch, "live").src, label: `Sub-stream frame · ${timeLabel(frame.epoch)}` };
      const approximate = frame.kind === "recording" ? "around " : "";
      return { ...frame, src, fallback, label: `${frame.label} · ${approximate}${timeLabel(frame.epoch)}` };
    }));
    const gallery = incidentGalleryFrames(incident).map((frame) => ({ ...frame, src: `${frame.event.snapshot_url}?width=1280&quality=85`, label: `${frame.label} · ${formatDateTime(frame.event.created_at, zone)}` }));
    return [...existing, ...gallery].filter((frame) => (seen.has(frame.src) ? false : (seen.add(frame.src), true)));
  }, [incident, observations, revision, zone]);

  const ongoing = ["active", "new", "updated"].includes(notification?.state || incident?.state);
  const identityNames = notification?.people || (incident?.identities || []).map((item) => item?.name).filter(Boolean);
  // Names from face capture join the incident's own identities.
  const people = [...new Set([...identityNames, ...faceNames])];
  const builtTitle = incident ? incidentStoryTitle(incident, { cameraName, objects, people, fallback: notification?.summary || "" }) : "";
  // Only a wording of the current evidence replaces the built-in title.
  // A reopened (ongoing again) incident goes back to the built-in title.
  // The wording must describe this exact revision, so an old answer never
  // comes back while a new one is being asked for.
  const providerTitle = titleSettled && aiTitle?.id === titleId && aiTitle?.status === "ready" && aiTitle.current !== false
    && Number(aiTitle.revision) === Number(incident?.revision) ? aiTitle.title : "";
  const title = providerTitle || builtTitle;
  const cardNames = cardIdentities(objects, incident?.identities);
  const cameraCount = new Set((incident?.episodes || []).map((episode) => episode.camera_id).concat(incident?.episodes?.length ? [] : [incident?.camera_id])).size;
  const crumbDay = Number.isFinite(startEpoch) ? dayLabel(startEpoch, zone) : "";
  const timelineHref = recordingsHref({ cameraId: frameCamera, epoch: Number.isFinite(shownEpoch) ? shownEpoch : startEpoch, source: still.source || "main" });
  const segment = playback?.segments[playback.index];
  const hasDetails = Boolean(data && !missing);
  const errorNotice = error ? <div className="incident-detail-notice" role="alert"><span>{error}{data ? " Showing the last received details." : ""}</span><button onClick={() => setRetry((value) => value + 1)}><RefreshCw size={16} />Retry</button></div> : null;

  useLayoutEffect(() => {
    const top = topRef.current;
    if (!top) return undefined;
    const page = top.parentElement;
    const measure = () => page.style.setProperty("--story-top-height", `${top.getBoundingClientRect().height}px`);
    const observer = new ResizeObserver(measure);
    measure();
    observer.observe(top);
    return () => {
      observer.disconnect();
      page.style.removeProperty("--story-top-height");
    };
  }, [hasDetails]);

  useEffect(() => {
    if (focusId == null || !window.matchMedia("(max-width: 899px)").matches) return;
    const frame = frameElementRef.current;
    if (!frame) return;
    const topbarHeight = parseFloat(getComputedStyle(frame).getPropertyValue("--topbar-height")) || 58;
    if (frame.getBoundingClientRect().top < topbarHeight) frame.scrollIntoView({ block: "start", behavior: "instant" });
  }, [focusId, pin]);

  useEffect(() => {
    if (!title) return;
    document.title = `${title} · SurvNG`;
    if (focusTitleRef.current) {
      focusTitleRef.current = false;
      titleRef.current?.focus();
    }
  }, [title, titleId]);

  return <main className="incident-story-page">
    {!hasDetails ? errorNotice : null}
    {missing ? <section className="incident-detail-empty"><ImageOff size={36} /><h1>Incident unavailable</h1><p>The incident may have expired or been removed.</p><button onClick={() => setRetry((value) => value + 1)}>Try again</button><a href={appUrl("/incidents")}>All incidents</a></section>
      : !data ? (!error && <section className="incident-detail-empty" aria-busy="true">Loading incident…</section>)
        : <>
          <div className="story-top" ref={topRef}>
            {errorNotice}
            <header className="story-header">
              <div className="story-heading">
                <nav className="story-crumb" aria-label="Breadcrumb">
                  {review ? <span><span className="story-wide-label">Review mode · </span>{position.index >= 0 ? `${position.index + 1} of ${position.total} · ` : ""}{crumbDay}</span>
                    : <><a href={appUrl("/incidents")}>Incidents</a><span aria-hidden="true"> / </span><span>{crumbDay}</span></>}
                </nav>
                <h1 ref={titleRef} tabIndex={-1} title={providerTitle ? `AI-worded title. Built-in: ${builtTitle}` : undefined}>{title}</h1>
                <p className="story-meta">
                  <time dateTime={incident.start_at}>{formatTimeOnly(startEpoch, zone)}</time> – <time dateTime={incident.end_at}>{formatTimeOnly(endEpoch, zone)}</time>
                  <span>· {formatSpan(endEpoch - startEpoch)}{cameraCount > 1 ? ` · ${cameraCount} cameras` : ""}</span>
                  {ongoing ? <span className="story-status">Ongoing</span> : null}
                  {aiTitle?.id === titleId && aiTitle?.status === "failed" ? <span className="story-ai-note" title={aiTitle.message || "AI title unavailable"}>· AI title unavailable</span> : null}
                </p>
              </div>
              <div className="story-header-actions">
                <button type="button" className="story-nav-button" aria-label="Previous" disabled={!position.previous} onClick={() => navigate(position.previous)}><ChevronLeft size={16} aria-hidden="true" /><span>Previous</span></button>
                <button type="button" className="story-nav-button" aria-label="Next" disabled={!position.next} onClick={() => navigate(position.next)}><span>Next</span><ChevronRight size={16} aria-hidden="true" /></button>
                <a className="story-button" aria-label="View in Timeline" href={timelineHref}><span className="story-wide-label">View in </span>Timeline</a>
                {ongoing ? <a className="story-button" href={appUrl(`/?camera=${encodeURIComponent(incident.camera_id)}`)}><Camera size={16} />Live view</a> : null}
                {canCorrectIncident ? <button type="button" className="story-review-toggle" aria-label={review ? "Exit review" : "Review mode"} aria-pressed={review} onClick={toggleReview}>{review ? "Exit review" : <>Review<span className="story-wide-label"> mode</span></>}</button> : null}
              </div>
            </header>

            {review ? <section className="story-review-bar" aria-label="Review">
              <div className="story-review-progress">
                <span>{position.remaining} left {day === localDay(Date.now() / 1000, zone) ? "today" : `on ${crumbDay}`}<span className="story-wide-label"> · J and K move without deciding</span></span>
                <div className="story-progress" role="img" aria-label={`${position.decided} of ${position.total} decided`}><div style={{ width: `${position.total ? 100 * position.decided / position.total : 0}%` }} /></div>
              </div>
              {VERDICTS.map((item) => {
                const current = position.index >= 0 && queue?.items[position.index]?.verdict === item.verdict;
                return <button key={item.verdict} type="button" className={`story-verdict ${item.verdict}`} aria-pressed={current} disabled={reviewBusy} onClick={() => void decide(item.verdict)} aria-label={item.label} aria-keyshortcuts={item.key}>
                  {item.label}<kbd>{item.key}</kbd></button>;
              })}
              {reviewError ? <p role="alert" className="story-review-message">{reviewError}</p> : null}
            </section> : null}
          </div>

          <div className="story-layout">
            <section ref={frameElementRef} className={`story-frame${playback ? " playing" : ""}`} aria-label="Incident evidence">
              <div className="story-frame-media">{/* Media alone in its slot: the player is remounted per camera. */}
              {playback && segment ? <IncidentRecordingPlayer key={playback.key} cameraId={segment.cameraId} startEpoch={segment.start} endEpoch={segment.end}
                initialEpoch={playback.initialEpoch} timeZone={zone} autoPlay onPlayhead={setCursor} onEnded={playbackEnded} onClose={stopPlayback}
                onUnavailable={() => { if (playback.index + 1 < playback.segments.length) playbackEnded(true); }} />
                : <StillFrame still={still} anchor={anchor} boxes={boxes} selectedId={focusId} legacyBoxes={legacyBoxes}
                  label={`Incident at ${cameraName(frameCamera)}`} onCover={!isCover ? () => selectObject(null) : null} />}
              </div>
              <div className="story-frame-chips">
                <span className="story-frame-chip strong">{cameraName(frameCamera)}</span>
                <span className="story-frame-chip"><time dateTime={Number.isFinite(shownEpoch) ? new Date(shownEpoch * 1000).toISOString() : undefined}>{formatTimeOnly(shownEpoch, zone)}</time> · {playback ? (selected ? `Following ${objectName(selected)}` : "Playing incident") : mode}</span>
              </div>
              <span className="sr-only" role="status" aria-live="polite">{notice?.text || ""}</span>
              {notice ? <div key={notice.key} className="story-switched" aria-hidden="true">{notice.text}</div> : null}
              {!playback ? <button type="button" className="story-play" disabled={!canPlay} onClick={() => startFollowing()}>
                <Play size={16} fill="currentColor" />{selected ? `Follow ${objectName(selected)}` : "Play incident"}</button> : null}
            </section>
            {selected ? <aside className="story-aside" aria-label="Selected sighting">
              {selected.kind === "uncertain" || correcting
                  ? <CorrectPanel incident={incident} object={selected} objects={objects} cameraName={cameraName} canCorrect={canCorrectIncident}
                    timeZone={zone} onCorrected={(detail, keepId) => {
                      if (currentIdRef.current === currentId) {
                        setCorrecting(true);
                        if (keepId) { setSelectedId(keepId); setNotice({ text: "Objects associated", key: Date.now() }); }
                      }
                      updateIncident(detail, currentId);
                    }} onDone={() => setCorrecting(false)} />
                  : <PathPanel object={selected} cameraName={cameraName} timeZone={zone} activeCameraId={frameCamera} canCorrect={canCorrectIncident}
                    onCorrect={() => setCorrecting(true)} onOpenEvent={async (eventId) => {
                      if (!eventId) return;
                      const from = currentId;
                      const response = await fetch(`/api/incidents/by-event/${encodeURIComponent(eventId)}`).catch(() => null);
                      const target = response?.ok ? await response.json().catch(() => null) : null;
                      if (currentIdRef.current !== from) return;
                      if (target?.incident_id || target?.id) navigate(target.incident_id || target.id, false);
                      else setNotice({ text: "Could not open that incident. Try again.", key: Date.now() });
                    }} />}
            </aside> : null}
            <IncidentStoryline line={line} cameraName={cameraName} selectedId={focusId} cursor={Number.isFinite(shownEpoch) ? shownEpoch : null} timeZone={zone} onBar={jumpToBar} />
            {objects.length ? <section className="story-sightings" aria-label="Sightings">
              <div className="story-sightings-heading">
                <h2 className="story-aside-title" ref={sightingsHeadingRef} tabIndex={-1}>Sightings ({objects.length})</h2>
                {selected ? <button className="story-link-button" type="button" onClick={() => {
                  selectObject(null);
                  sightingsHeadingRef.current?.focus({ preventScroll: true });
                }}>Clear selection</button> : null}
              </div>
              {incident.continuity_uncertain ? <p className="story-muted small">Some sightings may show the same subject.</p> : null}
              <ObjectCards objects={objects} selectedId={focusId} cameraName={cameraName} timeZone={zone} identities={cardNames} objectNames={faceObjects} onSelect={selectObject} />
            </section> : null}
            {!selected ? <aside className="story-aside" aria-label="Incident details">
              <WhyPanel incident={incident} cameraName={cameraName} hasSightings={objects.length > 0} />
            </aside> : null}
          </div>

          <section className="story-more">
            <IncidentPeople incidentId={incident.incident_id || currentId} eventIds={incident.event_ids || (incident.events || []).map((event) => event.id)} revision={incident.revision} onNames={setFaceNames} onObjects={setFaceObjects} />
            <details>
              <summary>Evidence images ({evidenceFrames.length})</summary>
              <div className="incident-detail-frames">
                {evidenceFrames.map((frame, index) => <EvidenceImage key={frame.src} frame={frame} onClick={() => setSelectedFrame({ frames: evidenceFrames, index })} />)}
              </div>
              {!evidenceFrames.length ? <p>No evidence images are available.</p> : <p className="incident-detail-hint">Frames from expired recordings may be unavailable.</p>}
            </details>
            {incident.speech ? <details><summary>Speech</summary><IncidentSpeechPanel incident={incident} timeZone={zone} /></details> : null}
            <details>
              <summary>Activity, alert policy and corrections</summary>
              <IncidentScenePanel compact key={incident.incident_id || incident.id} incident={incident} timeZone={zone} canCorrectIncident={canCorrectIncident}
                cameraNameById={{ get: cameraName }} onChanged={(detail) => updateIncident(detail, currentId)}
                onSelectEpisode={(episodeId) => { selectObject(null); startPlayback(segmentsFor(null).filter((item) => item.episodeId === episodeId)); }}
                activeEpisodeId={segment?.episodeId} onStopPlayback={stopPlayback}
                onNextEpisode={playback && playback.segments.slice(playback.index + 1).some((item) => item.episodeId && item.episodeId !== segment?.episodeId) ? () => {
                  const all = segmentsFor(null);
                  const next = all.findIndex((item, index) => index > all.findIndex((candidate) => candidate.episodeId === segment?.episodeId) && item.episodeId !== segment?.episodeId);
                  if (next >= 0) startPlayback(all, next, null, segment?.cameraId);
                } : null} />
            </details>
            <a className="story-link" href={appUrl(canonicalIncidentHref(incident))}>Open in the incident list</a>
          </section>
          {selectedFrame ? <EvidenceViewer frames={selectedFrame.frames} selected={selectedFrame.index} onClose={() => setSelectedFrame(null)} /> : null}
        </>}
  </main>;
}
