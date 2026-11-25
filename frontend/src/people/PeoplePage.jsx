import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleAlert, EyeOff, RotateCcw, ScanFace, UserCheck, UserMinus, Users, X } from "lucide-react";
import { appUrl, fetch, recordingsHref } from "../shared/api.js";
import { formatDateTime } from "../shared/format.js";

const PAGE_SIZE = 48;
const MAX_REQUEST = 200; // the API's largest page
const REFRESH_MS = 30000;
const TABS = [
  ["unnamed", "To name", ScanFace],
  ["named", "People", Users],
  ["ignored", "Ignored", EyeOff],
];

function faceUrl(faceId) {
  return appUrl(`/api/face-captures/${faceId}/image`);
}

function apiDetail(payload, fallback) {
  const detail = payload?.detail;
  return typeof detail === "string" && detail ? detail : fallback;
}

async function call(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401 || response.status === 403) {
    throw Object.assign(new Error("Only administrators can see and name people."), { forbidden: true });
  }
  if (!response.ok) throw new Error(apiDetail(payload, "The request failed."));
  return payload;
}

async function loadGroups(status, limit) {
  // Fetched in API-sized pages: scrolling grows the limit past what one request may ask for.
  let listing = null;
  for (let offset = 0; offset < limit; offset += MAX_REQUEST) {
    const page = await call(`/api/face-groups?status=${status}&limit=${Math.min(MAX_REQUEST, limit - offset)}&offset=${offset}`);
    listing = listing ? { ...page, groups: [...listing.groups, ...page.groups] } : page;
    if (page.groups.length < Math.min(MAX_REQUEST, limit - offset)) break;
  }
  return listing;
}

function cameraNames(ids, cameras) {
  const names = new Map((cameras || []).map((camera) => [camera.id, camera.name]));
  return (ids || []).map((id) => names.get(id) || id).join(", ");
}

function Faces({ ids, size = "small" }) {
  return <div className={`people-faces ${size}`}>
    {(ids || []).map((id) => <img key={id} src={faceUrl(id)} alt="" loading="lazy" />)}
  </div>;
}

function NameForm({ names, busy, onName, label = "Name" }) {
  const [value, setValue] = useState("");
  return <form className="people-name-form" onSubmit={(event) => {
    event.preventDefault();
    const name = value.trim();
    if (name) onName(name).then((ok) => { if (ok !== false) setValue(""); });
  }}>
    <input value={value} onChange={(event) => setValue(event.target.value)} placeholder="Who is this?" list="people-names" maxLength={80} aria-label="Name" disabled={busy} />
    <button type="submit" className="primary" disabled={busy || !value.trim()}><UserCheck size={14} />{label}</button>
  </form>;
}

function GroupCard({ group, cameras, names, busy, onName, onIgnore, onOpen, timeZone }) {
  return <article className="people-group-card">
    <button type="button" className="people-group-open" onClick={onOpen} aria-label="Open this group">
      <Faces ids={group.samples} />
    </button>
    <div className="people-group-body">
      <strong>{group.person_name || (group.ignored ? "Ignored" : "Unnamed")}</strong>
      <small>{group.faces} {group.faces === 1 ? "face" : "faces"} · {group.events} {group.events === 1 ? "visit" : "visits"} · {cameraNames(group.cameras, cameras)}</small>
      <small>Last seen {formatDateTime(group.last_epoch, timeZone)}</small>
      {group.suggestion ? <button type="button" className="people-suggestion" disabled={busy} onClick={() => onName(group.suggestion.name)}>
        <UserCheck size={14} />This is {group.suggestion.name}?
      </button> : null}
      {!group.person_id && !group.ignored ? <NameForm names={names} busy={busy} onName={onName} /> : null}
      <div className="people-group-actions">
        {!group.ignored ? <button type="button" disabled={busy} onClick={() => onIgnore(true)}><EyeOff size={14} />Ignore</button>
          : <button type="button" disabled={busy} onClick={() => onIgnore(false)}><RotateCcw size={14} />Not ignored</button>}
        <button type="button" onClick={onOpen}>Details</button>
      </div>
    </div>
  </article>;
}

function Visits({ visits, cameras, timeZone }) {
  if (!visits?.length) return null;
  return <section className="people-visits">
    <h3>Visits</h3>
    <ul>
      {visits.map((visit) => <li key={visit.event_id}>
        <img src={faceUrl(visit.best_face)} alt="" loading="lazy" />
        <span>{formatDateTime(visit.epoch, timeZone)}<small>{cameraNames([visit.camera_id], cameras)} · {visit.faces} {visit.faces === 1 ? "face" : "faces"}</small></span>
        <a href={recordingsHref({ cameraId: visit.camera_id, epoch: visit.epoch, eventId: visit.event_id })}>Recording</a>
      </li>)}
    </ul>
  </section>;
}

function GroupDetail({ groupId, cameras, names, timeZone, onClose, onChanged }) {
  const [group, setGroup] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      setGroup(await call(`/api/face-groups/${groupId}`));
      setError("");
    } catch (reason) {
      setError(reason.message);
    }
  }, [groupId]);
  useEffect(() => { void load(); }, [load]);

  async function act(path, body) {
    setBusy(true);
    try {
      await call(path, { method: "POST", body });
      await onChanged();
      await load();
      return true;
    } catch (reason) {
      setError(reason.message);
      return false;
    } finally {
      setBusy(false);
    }
  }

  return <aside className="people-detail" aria-label="Face group">
    <header>
      <h2>{group?.person_name || "Unnamed group"}</h2>
      <button type="button" className="icon-button" onClick={onClose} aria-label="Close"><X size={16} /></button>
    </header>
    {error ? <p className="settings-help" role="alert"><CircleAlert size={14} /> {error}</p> : null}
    {group ? <>
      <p className="settings-help">{group.faces} faces from {group.events} visits · {cameraNames(group.cameras, cameras)}</p>
      {group.person_id
        ? <button type="button" disabled={busy} onClick={() => act(`/api/face-groups/${groupId}/unname`)}><UserMinus size={14} />Not {group.person_name}</button>
        : <NameForm names={names} busy={busy} onName={(name) => act(`/api/face-groups/${groupId}/name`, { name })} />}
      <h3>Faces</h3>
      <p className="settings-help">Remove any face that is not this person; it moves to a group of its own.</p>
      <div className="people-face-grid">
        {(group.faces_list || []).map((face) => <figure key={face.id}>
          <img src={faceUrl(face.id)} alt="" loading="lazy" />
          <figcaption>
            <small>{formatDateTime(face.captured_epoch, timeZone)}</small>
            <button type="button" className="link-button" disabled={busy || group.faces <= 1} onClick={() => act(`/api/face-captures/${face.id}/remove`)}>Not this person</button>
          </figcaption>
        </figure>)}
      </div>
      <Visits visits={group.visits} cameras={cameras} timeZone={timeZone} />
    </> : !error ? <p className="settings-help">Loading…</p> : null}
  </aside>;
}

function PersonDetail({ personId, cameras, timeZone, onClose, onChanged, onOpenGroup, onOpenPerson }) {
  const [person, setPerson] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [rename, setRename] = useState("");
  const load = useCallback(async () => {
    try {
      const next = await call(`/api/face-people/${personId}`);
      setPerson(next);
      setRename(next.name);
      setError("");
    } catch (reason) {
      setError(reason.message);
    }
  }, [personId]);
  useEffect(() => { void load(); }, [load]);

  async function act(path, options) {
    setBusy(true);
    try {
      const result = await call(path, options);
      await onChanged();
      return result;
    } catch (reason) {
      setError(reason.message);
      return null;
    } finally {
      setBusy(false);
    }
  }

  return <aside className="people-detail" aria-label="Person">
    <header>
      <h2>{person?.name || "Person"}</h2>
      <button type="button" className="icon-button" onClick={onClose} aria-label="Close"><X size={16} /></button>
    </header>
    {error ? <p className="settings-help" role="alert"><CircleAlert size={14} /> {error}</p> : null}
    {person ? <>
      <form className="people-name-form" onSubmit={async (event) => {
        event.preventDefault();
        const result = await act(`/api/face-people/${personId}/rename`, { method: "POST", body: { name: rename } });
        // Renaming to another person's name joins this person into that one.
        if (result && result.id !== personId) onOpenPerson(result.id);
        else if (result) await load();
      }}>
        <input value={rename} onChange={(event) => setRename(event.target.value)} maxLength={80} aria-label="Name" disabled={busy} />
        <button type="submit" disabled={busy || !rename.trim() || rename.trim() === person.name}>Rename</button>
        <button type="button" className="danger" disabled={busy} onClick={async () => {
          if (!window.confirm(`Remove the name ${person.name}? Their faces go back to "To name".`)) return;
          if (await act(`/api/face-people/${personId}`, { method: "DELETE" })) onClose();
        }}>Remove name</button>
      </form>
      <h3>Face groups</h3>
      <div className="people-person-groups">
        {person.groups.map((group) => <div key={group.id} className="people-person-group">
          <button type="button" className="people-group-open" onClick={() => onOpenGroup(group.id)}><Faces ids={group.samples} /></button>
          <small>{group.faces} faces · {cameraNames(group.cameras, cameras)}</small>
          <button type="button" className="link-button" disabled={busy} onClick={async () => {
            if (!(await act(`/api/face-groups/${group.id}/unname`, { method: "POST" }))) return;
            // Unnaming a person's last group removes the person.
            if (person.groups.length <= 1) onClose();
            else await load();
          }}>Not {person.name}</button>
        </div>)}
      </div>
      <Visits visits={person.visits} cameras={cameras} timeZone={timeZone} />
    </> : !error ? <p className="settings-help">Loading…</p> : null}
  </aside>;
}

export function PeoplePage({ timeZone }) {
  const [tab, setTab] = useState("unnamed");
  const [listing, setListing] = useState({ groups: [], total: 0, summary: null, cameras: [] });
  const [people, setPeople] = useState([]);
  const [limit, setLimit] = useState(PAGE_SIZE);
  const [error, setError] = useState("");
  const [forbidden, setForbidden] = useState(false);
  const [busyId, setBusyId] = useState(null);
  const [openGroup, setOpenGroup] = useState(null);
  const [openPerson, setOpenPerson] = useState(null);
  const sentinel = useRef(null);
  const names = useMemo(() => people.map((person) => person.name), [people]);

  const load = useCallback(async () => {
    try {
      const [groups, peopleList] = await Promise.all([
        tab === "named" ? null : loadGroups(tab, limit),
        call("/api/face-people"),
      ]);
      if (groups) setListing(groups);
      else setListing((current) => ({ ...current, summary: peopleList.summary }));
      setPeople(peopleList.people || []);
      setError("");
    } catch (reason) {
      setForbidden(Boolean(reason.forbidden));
      setError(reason.message);
    }
  }, [tab, limit]);

  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    const timer = window.setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [load]);
  useEffect(() => { setLimit(PAGE_SIZE); }, [tab]);
  useEffect(() => {
    const element = sentinel.current;
    if (!element || tab === "named" || listing.groups.length >= listing.total) return undefined;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) setLimit((current) => current + PAGE_SIZE);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [tab, listing.groups.length, listing.total]);

  async function act(groupId, path, body) {
    setBusyId(groupId);
    try {
      await call(path, { method: "POST", body });
      await load();
      return true;
    } catch (reason) {
      setError(reason.message);
      return false;
    } finally {
      setBusyId(null);
    }
  }

  const summary = listing.summary;
  const counts = { unnamed: summary?.unnamed_groups, named: summary?.people, ignored: summary?.ignored_groups };
  if (forbidden) return <main className="people-page"><p className="settings-help" role="alert"><CircleAlert size={14} /> {error}</p></main>;
  return <main className="people-page">
    <datalist id="people-names">{names.map((name) => <option key={name} value={name} />)}</datalist>
    <header className="people-header">
      <h1>People</h1>
      <p className="settings-help">Faces captured from recorded person events, grouped by who they look like. Name a group once; later faces of that person are named for you. Typing a name that already exists joins the group to that person.</p>
      <nav className="people-mode-tabs" aria-label="People views">
        {TABS.map(([value, label, Icon]) => <button type="button" key={value} className={tab === value ? "active" : ""} aria-pressed={tab === value} onClick={() => { setTab(value); setOpenGroup(null); setOpenPerson(null); }}>
          <Icon size={15} />{label}{Number.isFinite(counts[value]) ? <span className="people-count">{counts[value]}</span> : null}
        </button>)}
      </nav>
    </header>
    {error ? <p className="settings-help" role="alert"><CircleAlert size={14} /> {error}</p> : null}
    <div className={`people-layout${openGroup || openPerson ? " with-detail" : ""}`}>
      <section className="people-list">
        {tab === "named"
          ? (people.length ? <div className="people-grid">
            {people.map((person) => <article key={person.id} className="people-person-card">
              <button type="button" className="people-group-open" onClick={() => { setOpenPerson(person.id); setOpenGroup(null); }}>
                {person.cover ? <img src={faceUrl(person.cover)} alt="" loading="lazy" /> : <Users size={32} />}
              </button>
              <strong>{person.name}</strong>
              <small>{person.faces} faces{person.last_epoch ? ` · last seen ${formatDateTime(person.last_epoch, timeZone)}` : ""}</small>
            </article>)}
          </div> : <p className="settings-help">No one is named yet. Name a group in “To name”.</p>)
          : (listing.groups.length ? <div className="people-grid">
            {listing.groups.map((group) => <GroupCard key={group.id} group={group} cameras={listing.cameras} names={names} timeZone={timeZone}
              busy={busyId === group.id}
              onName={(name) => act(group.id, `/api/face-groups/${group.id}/name`, { name })}
              onIgnore={(ignored) => act(group.id, `/api/face-groups/${group.id}/ignore`, { ignored })}
              onOpen={() => { setOpenGroup(group.id); setOpenPerson(null); }} />)}
          </div> : <p className="settings-help">{tab === "unnamed"
            ? "Nothing to name yet. Faces appear here as face capture runs (Settings › Detection › Faces)."
            : "Nothing is ignored."}</p>)}
        {tab !== "named" && listing.groups.length < listing.total ? <div ref={sentinel} className="people-more">Loading more…</div> : null}
      </section>
      {openGroup ? <GroupDetail groupId={openGroup} cameras={listing.cameras} names={names} timeZone={timeZone}
        onClose={() => setOpenGroup(null)} onChanged={load} /> : null}
      {openPerson && !openGroup ? <PersonDetail personId={openPerson} cameras={listing.cameras} timeZone={timeZone}
        onClose={() => setOpenPerson(null)} onChanged={load} onOpenGroup={(id) => setOpenGroup(id)} onOpenPerson={setOpenPerson} /> : null}
    </div>
  </main>;
}
