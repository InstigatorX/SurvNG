import React, { useEffect, useState } from "react";
import { Mic, Play } from "lucide-react";
import { fetch, recordingsHref } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";
import { incidentSpeechLabel, speechSegmentPlaceholder, speechSegmentRecordingContext, speechSegmentsQuery } from "../speechSettings.mjs";

export function IncidentSpeechBadge({ speech }) {
  const label = incidentSpeechLabel(speech);
  if (!label) return null;
  return <span className="pill speech-pill" title={label}><Mic size={12} />{speech.segments > 1 ? speech.segments : ""}<span className="sr-only">{label}</span></span>;
}

// Transcript text comes from the admin-only speech API, never from the
// incident payload, which integrations can read with a read-scope token.
export function IncidentSpeechPanel({ incident, timeZone }) {
  const speech = incident?.speech;
  const windows = speech?.windows || [];
  const windowKey = JSON.stringify(windows);
  const [segments, setSegments] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!windows.length) return undefined;
    const controller = new AbortController();
    setSegments(null);
    setError("");
    (async () => {
      try {
        const pages = await Promise.all(windows.map(async (window) => {
          const query = speechSegmentsQuery({ cameraId: window.camera_id, limit: 200 });
          const params = new URLSearchParams(query);
          params.set("start", window.start_at);
          params.set("end", window.end_at);
          const response = await fetch(`/api/speech/segments?${params}`, { signal: controller.signal });
          if (response.status === 401 || response.status === 403) throw new Error("Transcripts are visible to administrators only.");
          if (!response.ok) throw new Error("Transcript is unavailable.");
          return (await response.json()).segments || [];
        }));
        const unique = new Map(pages.flat().map((segment) => [segment.id, segment]));
        setSegments([...unique.values()].sort((left, right) => String(left.started_at).localeCompare(String(right.started_at))));
      } catch (reason) {
        if (reason?.name !== "AbortError") setError(reason.message || "Transcript is unavailable.");
      }
    })();
    return () => controller.abort();
  }, [windowKey]);

  if (!windows.length) return null;
  return <section className="incident-speech-panel" aria-label="Speech during this incident">
    <h3><Mic size={15} />Speech <small>{incidentSpeechLabel(speech)}</small></h3>
    {error ? <p className="incident-speech-note">{error}</p> : null}
    {!error && segments === null ? <p className="incident-speech-note">Loading transcript…</p> : null}
    {segments?.length ? <ol>
      {segments.map((segment) => {
        const context = speechSegmentRecordingContext(segment);
        const text = segment.text?.trim();
        return <li key={segment.id}>
          <time dateTime={segment.started_at}>{formatDateTime(segment.started_at, timeZone)}</time>
          <p className={text ? "" : "empty"}>{text ? `“${text}”` : speechSegmentPlaceholder(segment)}</p>
          {context ? <a href={recordingsHref(context)}><Play size={13} />Play</a> : null}
        </li>;
      })}
    </ol> : null}
    {segments && !segments.length && !error ? <p className="incident-speech-note">The transcript has been removed by retention.</p> : null}
  </section>;
}
