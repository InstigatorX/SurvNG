import React, { useEffect, useRef, useState } from "react";
import { fetch } from "../shared/api.js";
import { formatTimeOnly } from "../shared/format.js";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { recordingReviewActive, recordingReviewClosed, recordingReviewMinute, recordingReviewSummary, recordingReviewUrl } from "../recordingReview.mjs";

// Explicitly requested reviews stay pinned while playback crosses minute
// boundaries. Camera/source changes or choosing another minute reset ownership.
export function RecordingReviewPanel({ cameraId, source, epoch, timeZone, onSeek }) {
  const currentStart = recordingReviewMinute(epoch);
  if (!cameraId || currentStart === null) return null;
  return <ReviewScope key={`${cameraId}:${source}`} {...{ cameraId, source, currentStart, timeZone, onSeek }} />;
}

function ReviewScope({ currentStart, ...props }) {
  const [pinnedStart, setPinnedStart] = useState(null);
  const start = pinnedStart ?? currentStart;
  return <ReviewMinute key={start} {...props} start={start} onPin={() => setPinnedStart(start)} pinned={pinnedStart !== null} canFollow={currentStart !== start} onFollow={() => setPinnedStart(null)} />;
}

function ReviewMinute({ cameraId, source, start, timeZone, onSeek, onPin, pinned, canFollow, onFollow }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [requested, setRequested] = useState(false);
  const postRef = useRef(null);
  const sequenceRef = useRef(0);
  const mountedRef = useRef(true);
  const lastRequestRef = useRef(0);
  const admittedRequestRef = useRef(null);
  const url = recordingReviewUrl(cameraId, source, start);
  const closed = recordingReviewClosed(start);
  const active = recordingReviewActive(result?.state);

  useEffect(() => {
    mountedRef.current = true;
    const onVisibility = () => {
      if (!document.hidden) return;
      // An intentional return to this review can resume the lease with a click.
      setRequested(false);
      admittedRequestRef.current = null;
      postRef.current?.abort();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      mountedRef.current = false;
      sequenceRef.current += 1;
      postRef.current?.abort();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  async function request(method, signal, { explicit = false, retry = false } = {}) {
    const sequence = ++sequenceRef.current;
    try {
      const requestUrl = retry ? `${url}&retry=true` : method === "POST" && !explicit
        ? `${url}&expected_request_id=${encodeURIComponent(admittedRequestRef.current)}` : url;
      const response = await fetch(requestUrl, { method, signal });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `Review unavailable (${response.status})`);
      if (!mountedRef.current || signal.aborted || sequence !== sequenceRef.current) return;
      setResult(payload);
      setError("");
      if (method === "POST" && explicit) admittedRequestRef.current = payload.request_id ?? null;
      if (!explicit && admittedRequestRef.current !== null && payload.request_id !== admittedRequestRef.current) {
        admittedRequestRef.current = null;
        setRequested(false);
      }
      if (!recordingReviewActive(payload.state)) setRequested(false);
    } catch (reason) {
      if (!mountedRef.current || signal.aborted || sequence !== sequenceRef.current) return;
      setError(reason.message || "Could not load this review. Playback is still available.");
      if (method === "POST") setRequested(false);
    }
  }

  useVisiblePolling(async (signal) => {
    if (postRef.current) return;
    await request("GET", signal);
  }, 5000, closed);

  async function analyze(renew = false) {
    if (postRef.current || document.hidden || !closed) return;
    if (renew && (!admittedRequestRef.current || result?.request_id !== admittedRequestRef.current)) return;
    if (!renew) onPin();
    const controller = new AbortController();
    postRef.current = controller;
    setSubmitting(true);
    setRequested(true);
    lastRequestRef.current = Date.now();
    await request("POST", controller.signal, { explicit: !renew, retry: !renew && ["failed", "partial", "unavailable"].includes(result?.state) });
    if (postRef.current === controller) postRef.current = null;
    if (mountedRef.current) setSubmitting(false);
  }

  useVisiblePolling(async () => {
    if (Date.now() - lastRequestRef.current >= 30000) await analyze(true);
  }, 30000, requested && active, { immediate: false });

  if (result?.enabled === false) return null;
  const observations = Array.isArray(result?.observations) ? result.observations : [];
  const completed = result?.state === "sampled" || result?.state === "partial";
  const retryable = ["failed", "partial", "unavailable"].includes(result?.state);
  const canRequest = closed && result?.enabled === true && (result?.has_recordings !== false || retryable) && result?.state !== "sampled" && !submitting;
  return <section className="recording-review-panel" aria-label="Recordings-first review">
    <header><div><strong>Recordings-first review <small>Experimental</small></strong><p>{formatTimeOnly(start, timeZone)}–{formatTimeOnly(start + 60, timeZone)} · Selected minute</p></div>
      <button type="button" disabled={!canRequest || (requested && active)} onClick={() => void analyze()}>{submitting ? "Requesting…" : active ? requested ? "Review in progress" : "Continue this review" : retryable ? "Retry sampled review" : "Analyze this minute"}</button>
    </header>
    {pinned ? <p>This review stays on the selected minute while playback continues. <button type="button" disabled={!canFollow} onClick={onFollow}>Review current playback minute</button></p> : null}
    <p>Checks 12 frames, one every 5 seconds. This is a sample, not continuous tracking: brief activity can be missed. Playback does not wait for review.</p>
    <p>This view shows recordings, not the automatic incident list. Review results stay separate from incidents and alerts. Smart Search does not search every recording.</p>
    {!closed ? <p role="status">Choose an earlier minute; this recording window is still closing.</p> : <p role="status">{recordingReviewSummary(result)}{result ? ` · ${Number(result.sample_count) || 0} / ${Number(result.planned_samples) || 12} frames checked` : ""}</p>}
    {error ? <p role="alert">{error}</p> : null}
    {result?.message ? <p>{result.message}</p> : null}
    {retryable ? <p>Retry runs a new bounded sample check; it does not start automatically.</p> : null}
    {completed && !observations.length ? <p>No objects found in sampled frames. This does not rule out activity between samples.</p> : null}
    {(result?.missing_timestamps?.length || 0) > 0 ? <p>{result.missing_timestamps.length} sample times could not be checked; coverage is incomplete.</p> : null}
    {observations.length ? <ul aria-label="Sampled observations">{observations.map((observation, index) => <li key={`${observation.timestamp}:${observation.label}:${index}`}><button type="button" onClick={() => { onPin(); onSeek?.(Number(observation.timestamp)); }} disabled={!Number.isFinite(Number(observation.timestamp))}>Around {formatTimeOnly(Number(observation.timestamp), timeZone)} · {observation.label} · {Math.round(Number(observation.confidence || 0) * 100)}%</button></li>)}</ul> : null}
  </section>;
}
