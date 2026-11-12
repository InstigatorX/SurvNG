import React, { useEffect, useRef, useState } from "react";
import { CircleAlert, Mic, Play, Search } from "lucide-react";
import { appUrl, fetch, recordingsHref } from "../shared/api.js";
import { canonicalIncidentHref } from "../incidentScene.mjs";
import { formatDateTime } from "../shared/format.js";
import { speechLevelLabel, speechSegmentPlaceholder, speechSegmentRecordingContext, speechSegmentStatusLabel, speechSegmentsQuery } from "../speechSettings.mjs";

const PAGE_SIZE = 50;

export function SpeechSearchWorkspace({ cameras, cameraId, timeZone, initialQuery = "", disabledReason = "" }) {
  const [query, setQuery] = useState(initialQuery);
  const [submittedQuery, setSubmittedQuery] = useState(initialQuery);
  const [segments, setSegments] = useState([]);
  const [nextBeforeId, setNextBeforeId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestRef = useRef(null);

  async function load({ searchQuery = submittedQuery, append = false } = {}) {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setLoading(true);
    setError("");
    try {
      const params = speechSegmentsQuery({
        query: searchQuery,
        cameraId,
        limit: PAGE_SIZE,
        beforeId: append ? nextBeforeId : null,
      });
      const response = await fetch(`/api/speech/segments?${params}`, { signal: controller.signal });
      const payload = await response.json().catch(() => ({}));
      if (response.status === 401 || response.status === 403) throw new Error("Transcripts are visible to administrators only.");
      if (!response.ok) throw new Error(payload.detail || "Speech search failed.");
      const rows = payload.segments || [];
      setSegments((current) => (append ? [...current, ...rows] : rows));
      setNextBeforeId(payload.next_before_id ?? null);
      setSubmittedQuery(searchQuery);
      const urlParams = new URLSearchParams({ mode: "speech" });
      if (searchQuery) urlParams.set("q", searchQuery);
      if (cameraId) urlParams.set("camera", cameraId);
      window.history.replaceState(null, "", appUrl(`/search?${urlParams.toString()}`));
    } catch (reason) {
      if (reason?.name !== "AbortError") setError(reason.message || "Speech search failed.");
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setLoading(false);
      }
    }
  }

  // Recent speech is listed without a query; the camera filter re-runs it.
  useEffect(() => { void load(); }, [cameraId]);
  useEffect(() => () => requestRef.current?.abort(), []);

  function submit(event) {
    event?.preventDefault();
    void load({ searchQuery: query.trim() });
  }

  function reset() {
    setQuery("");
    void load({ searchQuery: "" });
  }

  const cameraName = (id) => cameras.find((camera) => camera.id === id)?.name || id;
  const heading = loading && !segments.length
    ? "Searching…"
    : submittedQuery
      ? `${segments.length}${nextBeforeId ? "+" : ""} ${segments.length === 1 && !nextBeforeId ? "transcript" : "transcripts"} matching “${submittedQuery}”`
      : "Recent speech";

  return <section className="semantic-search-workspace speech-search-workspace">
    <form onSubmit={submit} className="semantic-search-form"><Search size={20} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder='Words that were said, e.g. “package” or “front door”' aria-label="Search transcripts" autoFocus /><div className="semantic-search-actions"><button type="button" className="secondary" onClick={reset} disabled={loading || (!query && !submittedQuery)}>Reset</button><button type="submit" disabled={loading}>{loading ? "Searching…" : "Search"}</button></div></form>
    <div className="search-results-heading" aria-live="polite"><strong>{heading}</strong>{cameraId ? <small>{cameraName(cameraId)}</small> : <small>All cameras</small>}</div>
    {disabledReason ? <div className="semantic-search-error"><CircleAlert size={17} />{disabledReason}</div> : null}
    {error ? <div className="semantic-search-error"><CircleAlert size={17} />{error}</div> : null}
    <div className="speech-search-results">
      {segments.map((segment) => {
        const context = speechSegmentRecordingContext(segment);
        const name = cameraName(segment.camera_id);
        const when = formatDateTime(segment.started_at, timeZone);
        const text = segment.text?.trim();
        const placeholder = speechSegmentPlaceholder(segment);
        return <article key={segment.id} className={`speech-result ${segment.status}`} aria-label={`Speech at ${name}, ${when}`}>
          <div className="speech-result-meta"><Mic size={15} /><strong>{name}</strong><small>{when} · {Number(segment.duration_seconds || 0).toFixed(1)}s{speechLevelLabel(segment) ? ` · ${speechLevelLabel(segment)}` : ""}</small><span className={`speech-result-status ${segment.status}`}>{speechSegmentStatusLabel(segment.status)}</span></div>
          <p className={text ? "speech-result-text" : "speech-result-text empty"}>{text ? `“${text}”` : placeholder}</p>
          {context || segment.incident_id ? <nav aria-label={`Actions for speech at ${name}`}>
            {segment.incident_id ? <a href={appUrl(canonicalIncidentHref({ incident_id: segment.incident_id }))}>Open incident</a> : null}
            {context ? <a href={recordingsHref(context)}><Play size={14} />Play recording</a> : null}
          </nav> : null}
        </article>;
      })}
      {!loading && !error && !segments.length ? <div className="semantic-search-empty"><Mic size={28} /><strong>{submittedQuery ? "No transcripts match" : "No speech recorded yet"}</strong><span>{submittedQuery ? "Try fewer or different words, or choose All cameras." : "Speech appears here when a listening camera hears someone talk."}</span></div> : null}
      {nextBeforeId && !error ? <button type="button" className="secondary speech-load-more" onClick={() => void load({ append: true })} disabled={loading}>{loading ? "Loading…" : "Load more"}</button> : null}
    </div>
  </section>;
}
