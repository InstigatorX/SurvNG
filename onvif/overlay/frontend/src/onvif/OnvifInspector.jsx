import React, { useEffect, useMemo, useRef, useState } from "react";

function basePath() {
  const path = window.location.pathname.replace(/\/+$/, "");
  return path.endsWith("/onvif") ? path.slice(0, -"/onvif".length) : "";
}

const API = `${basePath()}/api/onvif-inspector`;

function fmtTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 3,
  });
}

function fmtRemaining(value) {
  if (value === null || value === undefined) return "—";
  const seconds = Math.max(0, Number(value) || 0);
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m ${rest}s`;
}

function stateGlyph(value) {
  if (value === true) return "●";
  if (value === false) return "○";
  return "·";
}

function classNameForState(value) {
  if (value === true) return "state state-on";
  if (value === false) return "state state-off";
  return "state state-unknown";
}

export default function OnvifInspector() {
  const [events, setEvents] = useState([]);
  const [snapshot, setSnapshot] = useState({ cameras: {} });
  const [selected, setSelected] = useState(null);
  const [camera, setCamera] = useState("");
  const [recognizedOnly, setRecognizedOnly] = useState(false);
  const [changesOnly, setChangesOnly] = useState(false);
  const [paused, setPaused] = useState(false);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const cursor = useRef(0);

  async function loadEvents() {
    const params = new URLSearchParams({
      after: String(cursor.current),
      limit: "500",
    });
    if (camera) params.set("camera", camera);
    if (recognizedOnly) params.set("recognized_only", "true");
    if (changesOnly) params.set("changes_only", "true");

    const response = await fetch(`${API}/events?${params.toString()}`, {
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`events HTTP ${response.status}`);
    const payload = await response.json();
    cursor.current = Number(payload.next || cursor.current);
    if (!paused && payload.events?.length) {
      setEvents((current) => [...current, ...payload.events].slice(-1000));
    }
  }

  async function loadState() {
    const response = await fetch(`${API}/state`, { cache: "no-store" });
    if (!response.ok) throw new Error(`state HTTP ${response.status}`);
    setSnapshot(await response.json());
  }

  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        await Promise.all([loadEvents(), loadState()]);
        if (!cancelled) setError("");
      } catch (err) {
        if (!cancelled) setError(String(err?.message || err));
      }
    }
    refresh();
    const interval = window.setInterval(refresh, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [camera, recognizedOnly, changesOnly, paused]);

  async function clearInspector() {
    const response = await fetch(`${API}/clear`, { method: "POST" });
    if (!response.ok) throw new Error(`clear HTTP ${response.status}`);
    const payload = await response.json();
    cursor.current = Number(payload.next || cursor.current);
    setEvents([]);
    setSelected(null);
    await loadState();
  }

  const cameraNames = useMemo(
    () => Object.keys(snapshot.cameras || {}).sort(),
    [snapshot]
  );

  const visibleEvents = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return events;
    return events.filter((event) =>
      [
        event.camera_id,
        event.topic,
        event.normalized_topic,
        event.classification,
        event.active === null ? "unknown" : String(event.active),
        ...(event.simple_items || []).flatMap((item) => [item.name, item.value]),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase()
        .includes(needle)
    );
  }, [events, search]);

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>ONVIF Inspector</h1>
          <div className="subtitle">Live PullPoint events and parser state</div>
        </div>
        <div className="top-actions">
          <span className={`live-indicator ${paused ? "paused" : ""}`}>
            {paused ? "PAUSED" : "● LIVE"}
          </span>
          <button onClick={() => setPaused((value) => !value)}>
            {paused ? "Resume" : "Pause"}
          </button>
          <button onClick={clearInspector}>Clear</button>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="controls">
        <label>
          Camera
          <select
            value={camera}
            onChange={(event) => {
              cursor.current = 0;
              setEvents([]);
              setCamera(event.target.value);
            }}
          >
            <option value="">All cameras</option>
            {cameraNames.map((name) => (
              <option value={name} key={name}>{name}</option>
            ))}
          </select>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={recognizedOnly}
            onChange={(event) => {
              cursor.current = 0;
              setEvents([]);
              setRecognizedOnly(event.target.checked);
            }}
          />
          Recognized only
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={changesOnly}
            onChange={(event) => {
              cursor.current = 0;
              setEvents([]);
              setChangesOnly(event.target.checked);
            }}
          />
          Changes only
        </label>
        <label className="search">
          Search
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="topic, class, state..."
          />
        </label>
      </section>

      <section className="panel">
        <h2>Camera state</h2>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Camera</th>
                <th>Link</th>
                <th>Subscription</th>
                <th>Motion</th>
                <th>Person</th>
                <th>Vehicle</th>
                <th>Animal</th>
                <th>Face</th>
                <th>Notifications</th>
                <th>Unknown</th>
                <th>Renew err</th>
                <th>Resubs</th>
              </tr>
            </thead>
            <tbody>
              {cameraNames.map((name) => {
                const item = snapshot.cameras[name];
                const cls = item.classes || {};
                return (
                  <tr key={name}>
                    <td className="camera">{name}</td>
                    <td>
                      <span className={item.connected ? "ok" : "bad"}>
                        {item.connected ? "● connected" : "○ down"}
                      </span>
                    </td>
                    <td>{fmtRemaining(item.subscription_lifetime_seconds)}</td>
                    {["motion", "person", "vehicle", "animal", "face"].map((kind) => (
                      <td key={kind} className={classNameForState(cls[kind]?.active)}>
                        {stateGlyph(cls[kind]?.active)}
                      </td>
                    ))}
                    <td>{item.notifications_received ?? 0}</td>
                    <td>{item.unrecognized_notifications ?? 0}</td>
                    <td>{item.renewal_errors ?? 0}</td>
                    <td>{item.resubscriptions ?? 0}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel event-panel">
        <h2>Live events <span className="count">{visibleEvents.length}</span></h2>
        <div className="event-layout">
          <div className="table-scroll event-list">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Camera</th>
                  <th>Topic</th>
                  <th>Class</th>
                  <th>State</th>
                  <th>Δ</th>
                </tr>
              </thead>
              <tbody>
                {[...visibleEvents].reverse().map((event) => (
                  <tr
                    key={event.seq}
                    className={`event-row ${selected?.seq === event.seq ? "selected" : ""} ${!event.recognized ? "unrecognized" : ""}`}
                    onClick={() => setSelected(event)}
                  >
                    <td>{fmtTime(event.received_at)}</td>
                    <td className="camera">{event.camera_id}</td>
                    <td className="topic-cell">
                      {event.normalized_topic || event.topic || "—"}
                    </td>
                    <td>{event.classification || "unknown"}</td>
                    <td className={classNameForState(event.active)}>
                      {event.active === null ? "?" : event.active ? "TRUE" : "FALSE"}
                    </td>
                    <td>{event.changed ? "●" : ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <aside className="detail">
            {selected ? (
              <>
                <h3>Event detail</h3>
                <dl>
                  <dt>Camera</dt><dd>{selected.camera_id}</dd>
                  <dt>Received</dt><dd>{selected.received_at}</dd>
                  <dt>Event time</dt><dd>{selected.event_at || "—"}</dd>
                  <dt>Topic</dt><dd className="mono">{selected.topic || "—"}</dd>
                  <dt>Normalized</dt><dd className="mono">{selected.normalized_topic || "—"}</dd>
                  <dt>Classification</dt><dd>{selected.classification || "unknown"}</dd>
                  <dt>State</dt><dd>{selected.active === null ? "unknown" : String(selected.active)}</dd>
                  <dt>Changed</dt><dd>{String(Boolean(selected.changed))}</dd>
                </dl>

                <h4>SimpleItem values</h4>
                {selected.simple_items?.length ? (
                  <table className="items">
                    <tbody>
                      {selected.simple_items.map((item, index) => (
                        <tr key={`${item.name}-${index}`}>
                          <td className="mono">{item.name}</td>
                          <td className="mono">{item.value}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : <div className="muted">None extracted</div>}

                <details>
                  <summary>Raw message XML</summary>
                  <pre>{selected.message_xml || "(empty)"}</pre>
                </details>
              </>
            ) : (
              <div className="empty-detail">Select an event to inspect it.</div>
            )}
          </aside>
        </div>
      </section>
    </main>
  );
}
