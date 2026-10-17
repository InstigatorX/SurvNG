import React, { useEffect, useRef, useState } from "react";
import { FaceFindingPanel } from "./FaceFindingPanel.jsx";
import { Crosshair, Play, ScanFace, ShieldCheck, UserPlus, X } from "lucide-react";
import { appUrl, recordingsHref, fetch } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";

export function FaceReviewDialog({ observation, people, timeZone, onClose, onUpdated }) {
  const [newName, setNewName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const dialogRef = useRef(null);
  const headingRef = useRef(null);

  useEffect(() => {
    setNewName("");
    setError("");
    requestAnimationFrame(() => headingRef.current?.focus());
  }, [observation.id]);

  useEffect(() => {
    function handleDialogKey(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = [...(dialogRef.current?.querySelectorAll('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])') || [])];
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", handleDialogKey);
    return () => window.removeEventListener("keydown", handleDialogKey);
  }, [onClose]);

  const observedEpoch = Number.isFinite(Number(observation.observed_at))
    ? Number(observation.observed_at)
    : new Date(observation.observed_at).getTime() / 1000;
  const timelineHref = recordingsHref({ cameraId: observation.camera_id, epoch: observedEpoch });

  async function assignPerson(nextPersonId) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/faces/observations/${observation.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ person_id: nextPersonId ? Number(nextPersonId) : null }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Could not update this face");
      }
      await onUpdated?.("", { advance: true, observationId: observation.id });
    } catch (requestError) {
      setError(requestError.message || "Could not update this face");
    } finally {
      setBusy(false);
    }
  }

  async function createPerson() {
    const name = newName.trim();
    if (!name) return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/faces/people", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, observation_id: observation.id }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Could not create this person");
      }
      await onUpdated?.(`${name} enrolled`, { advance: true, observationId: observation.id });
    } catch (requestError) {
      setError(requestError.message || "Could not create this person");
    } finally {
      setBusy(false);
    }
  }

  async function updateReference(pinned) {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/faces/observations/${observation.id}/reference`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pinned }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || "Could not update this reference");
      }
      await onUpdated?.(pinned ? "Reference pinned" : "Reference unpinned", { advance: false });
    } catch (requestError) {
      setError(requestError.message || "Could not update this reference");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="face-review-dialog" ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="face-review-title">
        <button type="button" className="overlay-close" onClick={onClose} aria-label="Close"><X size={22} /></button>
        <img src={appUrl(`/api/faces/observations/${observation.id}/crop.jpg?padding=0.45`)} alt="Selected face" />
        <div className="face-review-form">
          <div><strong id="face-review-title" ref={headingRef} tabIndex={-1}>{observation.person_name || "Unknown face"}</strong><span>{observation.camera_id} · {formatDateTime(observation.observed_at, timeZone)}</span></div>
          <div className="face-match-summary">
            <span>Face quality <strong>{observation.quality_score != null ? `${Math.round(Number(observation.quality_score) * 100)}%` : "Not scored"}</strong></span>
            {Number(observation.consensus?.candidate_count || 0) > 1 ? <span>Selected from <strong>{observation.consensus.candidate_count} incident frames</strong>{Number(observation.consensus?.agreement_count || 0) > 1 ? ` · ${observation.consensus.agreement_count} agreed on identity` : ""}</span> : null}
            {observation.match_details?.reference_ids?.length ? <span>Match supported by <strong>{observation.match_details.reference_ids.length} strongest references</strong></span> : null}
            {observation.match_details?.margin != null ? <span>Lead over next person <strong>{Math.round(Number(observation.match_details.margin) * 100)} points</strong></span> : null}
          </div>
          {observation.candidate_person_id ? <div className="face-enroll-row"><button type="button" disabled={busy} onClick={() => assignPerson(observation.candidate_person_id)}><ScanFace size={16} /> Confirm {observation.candidate_person_name} ({Math.round(Number(observation.candidate_confidence || 0) * 100)}%)</button><button type="button" className="subtle" disabled={busy} onClick={() => assignPerson(null)}><X size={16} /> Reject</button></div> : null}
          {observation.auto_identified && observation.person_id ? <div className="face-enroll-row"><button type="button" disabled={busy} onClick={() => assignPerson(observation.person_id)}><ShieldCheck size={16} /> Confirm automatic match</button></div> : null}
          <label>Assign to person<select value={observation.person_id || ""} disabled={busy} onChange={(event) => assignPerson(event.target.value)}><option value="">Unknown</option>{people.map((person) => <option value={person.id} key={person.id}>{person.name}</option>)}</select></label>
          {observation.person_id && observation.review_status === "confirmed" ? <button type="button" className="subtle" disabled={busy} onClick={() => updateReference(!observation.reference_pinned)}><ShieldCheck size={16} /> {observation.reference_pinned ? "Unpin reference" : "Pin as reference"}</button> : null}
          <div className="face-enroll-row"><input value={newName} disabled={busy} onChange={(event) => setNewName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") createPerson(); }} placeholder="New person name" /><button type="button" onClick={createPerson} disabled={busy || !newName.trim()}><UserPlus size={16} /> Enroll</button></div>
          <nav className="face-evidence-links" aria-label="Face evidence links">
            {observation.event_id ? <a href={appUrl(`/incidents?event_ids=${observation.event_id}`)}>Open incident</a> : null}
            {observation.camera_id && Number.isFinite(observedEpoch) ? <a href={timelineHref}><Play size={14} />View in Timeline</a> : null}
          </nav>
          {error ? <span className="save-status error">{error}</span> : null}
        </div>
      </section>
    </div>
  );
}

export function FacesPage({ timeZone, onAssistantContextChange }) {
  useEffect(() => {
    onAssistantContextChange?.({
      page: "people",
      camera_id: "",
      incident_event_id: null,
      filters: { workflow: "face-finding" },
    });
  }, [onAssistantContextChange]);

  return (
    <main className="faces-page faces-mode-finding">
      <header className="faces-commandbar">
        <div><Crosshair size={18} /><span><strong>People</strong><small>Mark faces in these crops, then export them to train this install&apos;s detector</small></span></div>
      </header>
      <FaceFindingPanel timeZone={timeZone} />
    </main>
  );
}
