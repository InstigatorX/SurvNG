import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ChevronLeft,
  ChevronRight,
  Gauge,
  Play,
  RefreshCw,
  ScanFace,
  ShieldCheck,
  Trash2,
  UserPlus,
  Users,
  Workflow,
  X,
} from "lucide-react";
import { nextFaceReviewObservation } from "../faceReview.mjs";
import { createConditionalJsonClient } from "../conditionalJson.mjs";
import { PEOPLE_REVIEW_FILTERS, PEOPLE_WORKSPACE_MODES, peopleObservationRequestPlan, peopleWorkspaceSearch, readPeopleWorkspaceQuery } from "../peopleWorkspace.mjs";
import { appUrl, recordingsHref, fetch } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";
import { isMobileViewport } from "../shared/hooks.js";
import { useRuntimeState } from "../shared/runtimeState.jsx";

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
  const runtimeState = useRuntimeState();
  const initialPeopleQuery = useMemo(() => readPeopleWorkspaceQuery(window.location.search), []);
  const [people, setPeople] = useState([]);
  const [observations, setObservations] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [status, setStatus] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [clusterMembers, setClusterMembers] = useState([]);
  const [clusterBusy, setClusterBusy] = useState(false);
  const [clusterLoadError, setClusterLoadError] = useState("");
  const [clusterMemberError, setClusterMemberError] = useState("");
  const [calibration, setCalibration] = useState(null);
  const [calibrating, setCalibrating] = useState(false);
  const [filter, setFilter] = useState(initialPeopleQuery.status);
  const [mode, setMode] = useState(initialPeopleQuery.mode);
  const [cameraId, setCameraId] = useState(initialPeopleQuery.cameraId);
  const [personId, setPersonId] = useState(initialPeopleQuery.personId);
  const [selected, setSelected] = useState(null);
  const [notice, setNotice] = useState("");
  const [loadError, setLoadError] = useState("");
  const [exactFaceError, setExactFaceError] = useState("");
  const [requestedFaceId, setRequestedFaceId] = useState(initialPeopleQuery.faceId);
  const [selectedClusterId, setSelectedClusterId] = useState(initialPeopleQuery.clusterId);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(initialPeopleQuery.page);
  const [totalObservations, setTotalObservations] = useState(0);
  const peopleLoadSequence = useRef(0);
  const faceLoadSequence = useRef(0);
  const clusterLoadSequence = useRef(0);
  const clusterMemberLoadSequence = useRef(0);
  const stableDirectoryClient = useRef(null);
  if (!stableDirectoryClient.current) {
    stableDirectoryClient.current = createConditionalJsonClient(fetch);
  }
  const faceReviewTriggerRef = useRef(null);
  const faceReviewPanelRef = useRef(null);
  const initialFaceIdRef = useRef(initialPeopleQuery.faceId);
  const faceFiltersMountedRef = useRef(false);
  const peopleHistoryWriteRef = useRef(true);
  const pageSize = isMobileViewport() ? 24 : 48;
  const pageCount = Math.max(1, Math.ceil(totalObservations / pageSize));
  const modeCount = mode === "clusters" ? clusters.length : mode === "review" ? observations.length : totalObservations;
  const modeCountLabel = mode === "clusters" ? "clusters" : mode === "review" ? "queued" : "faces";

  useEffect(() => {
    onAssistantContextChange?.({
      page: "people",
      camera_id: selected?.camera_id || cameraId,
      incident_event_id: Number(selected?.event_id) || null,
      filters: { status: filter, person_id: personId },
    });
  }, [cameraId, filter, mode, onAssistantContextChange, personId, selected?.camera_id, selected?.event_id]);

  async function loadExactFace(faceId) {
    const requestedId = String(faceId || "");
    if (!requestedId) return;
    setExactFaceError("");
    try {
      const response = await fetch(`/api/faces/observations/${requestedId}`);
      if (response.status === 404) {
        setRequestedFaceId("");
        setNotice("That face observation is no longer available.");
        return;
      }
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Unable to open this face observation");
      setRequestedFaceId("");
      setSelected(payload);
    } catch (error) {
      setExactFaceError(error.message || "Unable to open this face observation");
    }
  }

  async function loadPeople() {
    const sequence = ++peopleLoadSequence.current;
    try {
      const payload = await stableDirectoryClient.current.get(
        "/api/faces/people",
        "Unable to load enrolled people",
      );
      if (sequence === peopleLoadSequence.current) setPeople(payload);
      return payload;
    } catch (error) {
      if (sequence === peopleLoadSequence.current) setLoadError(error.message || "Unable to load enrolled people");
      return null;
    }
  }

  async function load({ refreshPeople = false } = {}) {
    const sequence = ++faceLoadSequence.current;
    if (mode === "clusters") {
      if (refreshPeople) await loadPeople();
      if (sequence === faceLoadSequence.current) setLoading(false);
      return [];
    }
    setLoading(true);
    try {
      const requestPlan = peopleObservationRequestPlan({
        mode, status: filter, cameraId, personId, page, pageSize,
      });
      const peoplePromise = refreshPeople ? loadPeople() : Promise.resolve(null);
      const [observationResponse, countResponse] = await Promise.all([
        fetch(requestPlan.observations),
        requestPlan.count ? fetch(requestPlan.count) : Promise.resolve(null),
      ]);
      if (!observationResponse.ok) throw new Error("Unable to load the face database");
      const [observationPayload, countPayload] = await Promise.all([
        observationResponse.json(),
        countResponse?.ok ? countResponse.json() : null,
        peoplePromise,
      ]);
      if (sequence !== faceLoadSequence.current) return;
      setLoadError("");
      setObservations(observationPayload);
      if (countPayload) setTotalObservations(Number(countPayload.total || 0));
      setNotice("");
      return observationPayload;
    } catch (error) {
      if (sequence === faceLoadSequence.current) setLoadError(error.message || "Unable to load faces");
      return null;
    } finally {
      if (sequence === faceLoadSequence.current) setLoading(false);
    }
  }

  useEffect(() => {
    void loadPeople();
    const requestedId = initialFaceIdRef.current;
    initialFaceIdRef.current = "";
    if (requestedId) void loadExactFace(requestedId);
    return () => { peopleLoadSequence.current += 1; };
  }, []);
  useEffect(() => {
    if (runtimeState?.cameras) {
      setCameras(runtimeState.cameras);
      return undefined;
    }
    const controller = new AbortController();
    fetch("/api/cameras", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((payload) => { if (!controller.signal.aborted && Array.isArray(payload)) setCameras(payload); })
      .catch(() => { });
    return () => controller.abort();
  }, [runtimeState?.cameras]);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/faces/status", { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((payload) => { if (!controller.signal.aborted && payload) setStatus(payload); })
      .catch(() => { });
    return () => controller.abort();
  }, []);

  async function loadClusters() {
    const sequence = ++clusterLoadSequence.current;
    setClusterLoadError("");
    try {
      const clusterPayload = await stableDirectoryClient.current.get(
        "/api/faces/unknown-clusters",
        "Unable to load unknown clusters",
      );
      if (sequence !== clusterLoadSequence.current) return null;
      setClusters(clusterPayload);
      return clusterPayload;
    } catch (error) {
      if (sequence === clusterLoadSequence.current) setClusterLoadError(error.message || "Unable to load unknown clusters");
      return null;
    }
  }

  async function loadClusterMembers(clusterId = selectedClusterId) {
    const requestedId = String(clusterId || "");
    if (!requestedId) {
      setClusterMembers([]);
      return [];
    }
    const sequence = ++clusterMemberLoadSequence.current;
    setClusterMemberError("");
    try {
      const response = await fetch(`/api/faces/unknown-clusters/${encodeURIComponent(requestedId)}/members?limit=200`);
      const payload = await response.json().catch(() => []);
      if (!response.ok) throw new Error(payload.detail || "Unable to load cluster members");
      if (sequence !== clusterMemberLoadSequence.current) return null;
      setClusterMembers(payload);
      return payload;
    } catch (error) {
      if (sequence === clusterMemberLoadSequence.current) setClusterMemberError(error.message || "Unable to load cluster members");
      return null;
    }
  }

  useEffect(() => {
    if (mode !== "clusters") void load();
    else {
      faceLoadSequence.current += 1;
      setLoading(false);
      setLoadError("");
    }
    return () => { faceLoadSequence.current += 1; };
  }, [filter, cameraId, mode, personId, page]);
  useEffect(() => {
    if (mode !== "clusters") return;
    void loadClusters();
    return () => { clusterLoadSequence.current += 1; };
  }, [mode]);
  useEffect(() => {
    if (mode !== "clusters" || !selectedClusterId) {
      clusterMemberLoadSequence.current += 1;
      setClusterMembers([]);
      setClusterMemberError("");
      return;
    }
    void loadClusterMembers(selectedClusterId);
    return () => { clusterMemberLoadSequence.current += 1; };
  }, [mode, selectedClusterId]);
  useEffect(() => {
    if (!faceFiltersMountedRef.current) {
      faceFiltersMountedRef.current = true;
      return;
    }
    setPage(0);
  }, [filter, cameraId, personId, mode]);
  useEffect(() => { if (page >= pageCount) setPage(Math.max(0, pageCount - 1)); }, [page, pageCount]);
  useEffect(() => {
    if (!peopleHistoryWriteRef.current) {
      peopleHistoryWriteRef.current = true;
      return;
    }
    const search = peopleWorkspaceSearch({ mode, status: filter, cameraId, personId, page, faceId: selected?.id || requestedFaceId, clusterId: selectedClusterId });
    window.history.replaceState(window.history.state, "", appUrl(`/people${search}`));
  }, [cameraId, filter, mode, page, personId, requestedFaceId, selected?.id, selectedClusterId]);
  useEffect(() => {
    function restorePeopleState() {
      const restored = readPeopleWorkspaceQuery(window.location.search);
      peopleHistoryWriteRef.current = false;
      setMode(restored.mode);
      setFilter(restored.status);
      setCameraId(restored.cameraId);
      setPersonId(restored.personId);
      setPage(restored.page);
      setRequestedFaceId(restored.faceId);
      setSelectedClusterId(restored.clusterId);
      if (restored.faceId) void loadExactFace(restored.faceId);
      else {
        setSelected(null);
        requestAnimationFrame(() => faceReviewTriggerRef.current?.isConnected && faceReviewTriggerRef.current.focus());
      }
    }
    window.addEventListener("popstate", restorePeopleState);
    return () => window.removeEventListener("popstate", restorePeopleState);
  }, []);

  function openFaceReview(observation, trigger) {
    faceReviewTriggerRef.current = trigger || null;
    const search = peopleWorkspaceSearch({ mode, status: filter, cameraId, personId, page, faceId: observation.id, clusterId: selectedClusterId });
    window.history.pushState({ ...(window.history.state || {}), survngPeopleFace: true }, "", appUrl(`/people${search}`));
    peopleHistoryWriteRef.current = false;
    setRequestedFaceId("");
    setSelected(observation);
  }

  function closeFaceReview() {
    if (window.history.state?.survngPeopleFace) {
      window.history.back();
      return;
    }
    setSelected(null);
    setRequestedFaceId("");
    requestAnimationFrame(() => {
      const target = faceReviewTriggerRef.current?.isConnected
        ? faceReviewTriggerRef.current
        : document.querySelector(".face-observation-card") || faceReviewPanelRef.current;
      target?.focus();
    });
  }

  async function deletePerson(person) {
    if (!window.confirm(`Delete ${person.name}? Their observations will return to Unknown.`)) return;
    try {
      const response = await fetch(`/api/faces/people/${person.id}`, { method: "DELETE" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        return setNotice(payload.detail || "Could not delete this person");
      }
      setPersonId("");
      await Promise.all([loadPeople(), load()]);
    } catch (error) {
      setNotice(error.message || "Could not delete this person");
    }
  }

  async function analyzeCalibration() {
    setCalibrating(true);
    try {
      const response = await fetch("/api/faces/calibration");
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not analyze face matching");
      setCalibration(payload);
    } catch (error) {
      setNotice(error.message || "Could not analyze face matching");
    } finally {
      setCalibrating(false);
    }
  }

  async function rebuildClusters() {
    setClusterBusy(true);
    try {
      const response = await fetch("/api/faces/unknown-clusters/rebuild", { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "Could not rebuild unknown clusters");
      setSelectedClusterId("");
      await loadClusters();
      setNotice("Unknown clusters rebuilt from current face evidence.");
    } catch (error) {
      setNotice(error.message || "Could not rebuild unknown clusters");
    } finally {
      setClusterBusy(false);
    }
  }

  return (
    <main className={`faces-page faces-mode-${mode}`}>
      <header className="faces-commandbar">
        <div><Users size={18} /><span><strong>People</strong><small>Review identities and maintain trusted references</small></span></div>
        <div className="faces-command-status">
          <span><strong>{people.length}</strong> enrolled</span>
          <span><strong>{modeCount}</strong> {modeCountLabel}</span>
          <span className={status?.recognition_ready ? "healthy" : "caution"}><i />{status?.recognition_ready ? "Recognition ready" : "Needs attention"}</span>
        </div>
      </header>
      <nav className="people-mode-tabs" aria-label="People workspace mode">
        {Object.entries(PEOPLE_WORKSPACE_MODES).map(([value, label]) => (
          <button key={value} type="button" className={mode === value ? "active" : ""} aria-pressed={mode === value} onClick={() => { setMode(value); setPage(0); setSelectedClusterId(""); }}>
            {value === "review" ? <ScanFace size={16} /> : value === "people" ? <Users size={16} /> : <Workflow size={16} />} {label}
          </button>
        ))}
      </nav>
      <aside className="faces-people-panel">
        <div className="faces-panel-heading">
          <div><h2>People</h2><span>{people.length} enrolled</span></div>
          <Users size={20} />
        </div>
        <button type="button" className={`face-person-row ${personId === "" ? "active" : ""}`} aria-pressed={personId === ""} onClick={() => { setPersonId(""); setPage(0); }}>
          <span className="face-avatar unknown"><ScanFace size={22} /></span>
          <span><strong>All faces</strong><small>{status?.observations || 0} observations</small></span>
        </button>
        <div className="face-person-list">
          {people.map((person) => (
            <div className={`face-person-row ${String(person.id) === personId ? "active" : ""}`} key={person.id}>
              <button type="button" className="face-person-select" aria-pressed={String(person.id) === personId} onClick={() => { setPersonId(String(person.id)); setPage(0); }}>
                {person.preview_observation_id
                  ? <img src={appUrl(`/api/faces/observations/${person.preview_observation_id}/crop.jpg`)} alt="" />
                  : <span className="face-avatar unknown"><ScanFace size={20} /></span>}
                <span><strong>{person.name}</strong><small>{person.usable_reference_count || 0}/{person.reference_count || 0} usable references · {person.observation_count} total{person.pinned_reference_count ? ` · ${person.pinned_reference_count} pinned` : ""}</small></span>
              </button>
              <button type="button" className="icon-button subtle" onClick={() => deletePerson(person)} title="Delete person" aria-label={`Delete ${person.name}`}><Trash2 size={15} /></button>
            </div>
          ))}
        </div>
      </aside>

      <section className="faces-review-panel" ref={faceReviewPanelRef} tabIndex={-1} aria-label={PEOPLE_WORKSPACE_MODES[mode]}>
        <div className="faces-toolbar">
          {mode === "clusters" ? <div className="cluster-toolbar-copy"><strong>Unknown clusters</strong><small>Group recurring unknown faces before enrolling a person.</small></div> : null}
          {mode === "clusters" ? <button type="button" className="subtle" onClick={rebuildClusters} disabled={clusterBusy}><RefreshCw size={16} className={clusterBusy ? "spinning" : ""} /> {clusterBusy ? "Rebuilding…" : "Rebuild clusters"}</button> : null}
          {mode === "people" ? <>
          <div className="faces-filter-group" role="group" aria-label="Face status">
            {Object.entries(PEOPLE_REVIEW_FILTERS).map(([value, label]) => (
              <button type="button" className={filter === value && !personId ? "active" : ""} aria-pressed={filter === value && !personId} key={value} onClick={() => { setPersonId(""); setFilter(value); setPage(0); }}>{label}</button>
            ))}
          </div>
          <select className="faces-mobile-person-filter" value={personId} onChange={(event) => { setPersonId(event.target.value); setPage(0); }} aria-label="Filter by person">
            <option value="">All people</option>
            {people.map((person) => <option value={person.id} key={person.id}>{person.name}</option>)}
          </select>
          <select value={cameraId} onChange={(event) => { setCameraId(event.target.value); setPage(0); }} aria-label="Filter by camera">
            <option value="">All cameras</option>
            {cameras.map((camera) => <option value={camera.id} key={camera.id}>{camera.name || camera.id}</option>)}
          </select>
          <span className="shown-bubble">{totalObservations} faces</span>
          </> : mode === "review" ? <div className="queue-toolbar-copy"><strong>Priority review queue</strong><small>Suggestions and uncertain matches that benefit most from a decision.</small></div> : null}
        </div>
        <div className="face-messages">
          {!status?.recognition_ready || status?.recognition?.pending > 0 || status?.recognition?.failed > 0 ? <div className="face-readiness"><Activity size={16} /><span>{status?.recognition_message || "Automatic recognition is not configured."}</span></div> : null}
          <details className="face-calibration">
            <summary><span><Gauge size={16} /><strong>Matching health</strong><small>{calibration?.ready ? `${Math.round(Number(calibration.rank_one_accuracy || 0) * 100)}% measured identity accuracy` : "Analyze confirmed reviews to measure recognition"}</small></span><ChevronRight size={16} /></summary>
            <div><span>{calibration?.message || "Measure your confirmed and rejected faces. Analysis only—no settings are changed."}</span>{calibration?.ready ? <strong>{Math.round(Number(calibration.rank_one_accuracy || 0) * 100)}% identity accuracy · suggest at {Math.round(Number(calibration.recommended?.suggestion_threshold || 0) * 100)}% · automatic at {Math.round(Number(calibration.recommended?.automatic_threshold || 0) * 100)}% with a {Math.round(Number(calibration.recommended?.automatic_margin || 0) * 100)}-point lead</strong> : null}<button type="button" className="subtle" disabled={calibrating} onClick={analyzeCalibration}>{calibrating ? "Analyzing..." : "Analyze matching"}</button></div>
          </details>
          {notice ? <div className="save-status" role="status">{notice}</div> : null}
          {loading && observations.length ? <div className="face-updating" role="status">Updating results…</div> : null}
          {loadError ? <div className="face-load-error" role="alert"><span>{loadError}</span><button type="button" onClick={() => void load()}>Retry</button></div> : null}
          {exactFaceError ? <div className="face-load-error" role="alert"><span>{exactFaceError}</span><button type="button" onClick={() => void loadExactFace(requestedFaceId)}>Retry face</button></div> : null}
        </div>
        {mode === "clusters" ? <div className="unknown-cluster-layout">
          <ul className="unknown-cluster-grid" aria-label="Unknown face clusters">
            {clusterLoadError ? <li className="face-load-error" role="alert"><span>{clusterLoadError}</span><button type="button" onClick={() => void loadClusters()}>Retry</button></li> : null}
            {!clusterLoadError && !clusters.length ? <li className="empty-state">No recurring unknown-face clusters are ready yet.</li> : clusters.map((cluster) => (
              <li key={cluster.cluster_id}><button type="button" className={`unknown-cluster-card ${String(cluster.cluster_id) === selectedClusterId ? "active" : ""}`} aria-pressed={String(cluster.cluster_id) === selectedClusterId} onClick={() => setSelectedClusterId(String(cluster.cluster_id))}>
                <span className="cluster-preview" aria-hidden="true">
                  {cluster.preview_observation_id ? <img src={appUrl(`/api/faces/observations/${cluster.preview_observation_id}/crop.jpg`)} alt="" onError={(event) => { event.currentTarget.hidden = true; }} /> : null}
                  <span className="face-avatar unknown"><ScanFace size={24} /></span>
                </span>
                <span><strong>{cluster.name || `Unknown person ${cluster.cluster_id}`}</strong><small>{cluster.observation_count || 0} sightings · {cluster.camera_count || 0} cameras</small></span>
              </button></li>
            ))}
          </ul>
          {selectedClusterId ? <div className="cluster-members"><div><strong>Cluster members</strong><span>{clusterMembers.length} face observations</span></div>{clusterMemberError ? <div className="face-load-error" role="alert"><span>{clusterMemberError}</span><button type="button" onClick={() => void loadClusterMembers()}>Retry</button></div> : <div className="face-observation-grid compact">{clusterMembers.map((observation) => <button type="button" className="face-observation-card" key={observation.id} onClick={(event) => openFaceReview(observation, event.currentTarget)}><img src={appUrl(`/api/faces/observations/${observation.id}/crop.jpg`)} alt="Review clustered unknown face" loading="lazy" /><span className="face-card-hud"><strong>Unknown face</strong><small>{observation.camera_id} · {formatDateTime(observation.observed_at, timeZone)}</small></span></button>)}</div>}</div> : null}
        </div> : <div className="face-observation-grid">
          {loading && !observations.length ? <div className="empty-state">Loading face observations...</div> : null}
          {!loading && !loadError && !observations.length ? <div className="empty-state">No faces match these filters.</div> : null}
          {observations.map((observation) => (
            <button type="button" className="face-observation-card" key={observation.id} aria-label={`Review ${observation.person_name || observation.candidate_person_name || "unknown face"} from ${observation.camera_id} at ${formatDateTime(observation.observed_at, timeZone)}`} onClick={(event) => openFaceReview(observation, event.currentTarget)}>
              <img src={appUrl(`/api/faces/observations/${observation.id}/crop.jpg`)} alt={observation.person_name || "Unknown face"} loading="lazy" />
              <span className="face-card-hud">
                <strong>{observation.person_name || (observation.candidate_person_name ? `Suggested: ${observation.candidate_person_name}` : "Unknown")}</strong>
                <small>{observation.camera_id} · {formatDateTime(observation.observed_at, timeZone)}</small>
              </span>
              <span className="face-confidence">{Number(observation.consensus?.candidate_count || 0) > 1 ? `Best of ${observation.consensus.candidate_count} · ` : ""}{observation.candidate_confidence != null ? `${Math.round(Number(observation.candidate_confidence) * 100)}% match` : `${Math.round(Number(observation.confidence || 0) * 100)}%`}</span>
            </button>
          ))}
        </div>}
        {mode === "people" ? <div className="faces-pagination">
          <button type="button" onClick={() => setPage((current) => Math.max(0, current - 1))} disabled={page === 0 || loading}><ChevronLeft size={16} /> Previous</button>
          <span>Page {Math.min(page + 1, pageCount)} of {pageCount}</span>
          <button type="button" onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))} disabled={page >= pageCount - 1 || loading}>Next <ChevronRight size={16} /></button>
        </div> : null}
      </section>

      {selected ? <FaceReviewDialog observation={selected} people={people} timeZone={timeZone} onClose={closeFaceReview} onUpdated={async (message, action = {}) => {
        const isClusterReview = mode === "clusters" && Boolean(selectedClusterId);
        const currentObservations = isClusterReview ? clusterMembers : observations;
        if (message) setNotice(message);
        // The selected cluster can lose members after enrollment or rejection.
        // Refresh its independent list before choosing the next review target.
        const refreshed = isClusterReview
          ? (await Promise.all([loadPeople(), loadClusters()]), await loadClusterMembers(selectedClusterId))
          : await load({ refreshPeople: true });
        const nextObservation = action.advance && refreshed
          ? nextFaceReviewObservation(action.observationId || selected.id, currentObservations, refreshed)
          : null;
        if (nextObservation) {
          setSelected(nextObservation);
        } else {
          setSelected(null);
          if (action.advance) setNotice(message || "Review complete");
          requestAnimationFrame(() => {
            const target = faceReviewTriggerRef.current?.isConnected
              ? faceReviewTriggerRef.current
              : document.querySelector(".face-observation-card") || faceReviewPanelRef.current;
            target?.focus();
          });
        }
      }} /> : null}
    </main>
  );
}
