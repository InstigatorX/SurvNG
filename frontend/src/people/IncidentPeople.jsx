import React, { useCallback, useEffect, useState } from "react";
import { EyeOff, UserCheck } from "lucide-react";
import { appUrl, fetch } from "../shared/api.js";

// Faces captured in an incident's events, grouped by person (face capture).
// Admin-only data: for anyone else the API refuses and nothing is shown.

function faceUrl(faceId) {
  return appUrl(`/api/face-captures/${faceId}/image`);
}

export async function fetchFacesForEvents(eventIds, signal) {
  const ids = [...new Set((eventIds || []).map(Number).filter((id) => Number.isInteger(id) && id > 0))];
  if (!ids.length) return { events: {}, groups: [] };
  const response = await fetch(`/api/face-captures/by-events?ids=${ids.slice(0, 500).join(",")}`, { signal });
  if (!response.ok) return null; // not an admin, or face capture unavailable
  return response.json();
}

/** Named people seen in each incident: { [incidentKey]: ["Name", ...] }. */
export function useIncidentFaceNames(incidents) {
  const [names, setNames] = useState({});
  const key = (incidents || []).map((incident) => `${incident.id}:${(incident.events || []).length}`).join("|");
  useEffect(() => {
    const controller = new AbortController();
    const eventsOf = new Map((incidents || []).map((incident) => [String(incident.id), (incident.events || []).map((event) => Number(event.id))]));
    fetchFacesForEvents([...eventsOf.values()].flat(), controller.signal).then((found) => {
      if (!found) return;
      const personOf = new Map(found.groups.filter((group) => group.person_name && !group.ignored).map((group) => [group.id, group.person_name]));
      const next = {};
      for (const [incidentId, eventIds] of eventsOf) {
        const people = new Set();
        for (const eventId of eventIds) for (const groupId of found.events[String(eventId)] || []) {
          if (personOf.has(groupId)) people.add(personOf.get(groupId));
        }
        if (people.size) next[incidentId] = [...people];
      }
      setNames(next);
    }).catch(() => {});
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return names;
}

export function FaceNameChips({ names }) {
  if (!names?.length) return null;
  return <span className="incident-face-names" aria-label="People seen">
    {names.map((name) => <span key={name} className="incident-face-name"><UserCheck size={12} />{name}</span>)}
  </span>;
}

function NameBox({ busy, onName }) {
  const [value, setValue] = useState("");
  return <form className="people-name-form" onSubmit={async (event) => {
    event.preventDefault();
    const name = value.trim();
    if (name && await onName(name)) setValue("");
  }}>
    <input value={value} onChange={(event) => setValue(event.target.value)} placeholder="Who is this?" maxLength={80} aria-label="Name" disabled={busy} />
    <button type="submit" disabled={busy || !value.trim()}><UserCheck size={14} />Name</button>
  </form>;
}

export function IncidentPeople({ eventIds, onNames }) {
  const [found, setFound] = useState(null);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState("");
  const idsKey = (eventIds || []).join(",");

  const load = useCallback(async (signal) => {
    const next = await fetchFacesForEvents(eventIds, signal).catch(() => null);
    setFound(next);
    onNames?.((next?.groups || []).filter((group) => group.person_name && !group.ignored).map((group) => group.person_name));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  async function act(groupId, path, body) {
    setBusy(groupId);
    setError("");
    try {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "That did not work.");
      await load();
      return true;
    } catch (reason) {
      setError(reason.message);
      return false;
    } finally {
      setBusy(null);
    }
  }

  const groups = (found?.groups || []).filter((group) => !group.ignored);
  if (!found || !groups.length) return null;
  return <section className="incident-people" aria-label="People in this incident">
    <h2 className="story-aside-title">People</h2>
    {error ? <p className="story-muted" role="alert">{error}</p> : null}
    {groups.map((group) => <div key={group.id} className="incident-people-group">
      <div className="incident-people-faces">
        {(group.incident_faces || group.samples || []).slice(0, 4).map((id) => <img key={id} src={faceUrl(id)} alt="" loading="lazy" />)}
      </div>
      {group.person_name
        ? <a className="incident-people-name" href={appUrl("/people")}><UserCheck size={14} />{group.person_name}</a>
        : <div className="incident-people-unnamed">
          <span className="story-muted">Unnamed person</span>
          {group.suggestion ? <button type="button" disabled={busy === group.id} onClick={() => act(group.id, `/api/face-groups/${group.id}/name`, { name: group.suggestion.name })}>
            <UserCheck size={14} />This is {group.suggestion.name}?</button> : null}
          <NameBox busy={busy === group.id} onName={(name) => act(group.id, `/api/face-groups/${group.id}/name`, { name })} />
          <button type="button" className="link-button" disabled={busy === group.id} onClick={() => act(group.id, `/api/face-groups/${group.id}/ignore`, { ignored: true })}>
            <EyeOff size={13} />Not a person</button>
        </div>}
    </div>)}
  </section>;
}
