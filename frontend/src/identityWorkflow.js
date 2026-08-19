const BASE = String(window.__SURVNG_BASE_PATH__ || "").replace(/\/+$/, "");
const api = (path) => `${BASE}${path}`;
const state = {
  tab: "review",
  people: [],
  health: [],
  review: [],
  confirmed: [],
  clusters: [],
  selected: new Set(),
  selectedPersonId: "",
  busy: false,
  message: "",
};

function isPeoplePath() {
  const pathname = window.location.pathname;
  const local = BASE && pathname.startsWith(BASE) ? pathname.slice(BASE.length) || "/" : pathname;
  return local === "/people" || local === "/faces";
}

async function jsonFetch(path, options = {}) {
  const response = await window.fetch(api(path), {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[char]));
}

function pct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${Math.round(n * 100)}%` : "—";
}

function score(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(3) : "—";
}

function badge(text, kind = "") {
  return `<span class="i13-badge ${kind}">${esc(text)}</span>`;
}

function cropUrl(id) {
  return api(`/api/faces/observations/${encodeURIComponent(id)}/crop.jpg`);
}

async function loadBase() {
  state.busy = true;
  render();
  try {
    const [people, health] = await Promise.all([
      jsonFetch("/api/faces/people"),
      jsonFetch("/api/faces/people/representation-health"),
    ]);
    state.people = people;
    state.health = health;
    if (!state.selectedPersonId && people.length) state.selectedPersonId = String(people[0].id);
    await loadTab(state.tab);
    state.message = "";
  } catch (error) {
    state.message = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

async function loadTab(tab) {
  state.tab = tab;
  state.busy = true;
  render();
  try {
    if (tab === "review") {
      state.review = await jsonFetch("/api/faces/review/queue?limit=60");
    } else if (tab === "confirmed") {
      state.confirmed = await jsonFetch("/api/faces/review/confirmed?limit=60");
    } else if (tab === "clusters") {
      state.clusters = await jsonFetch("/api/faces/unknown-clusters");
    }
    state.message = "";
  } catch (error) {
    state.message = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

async function bulkAction(action, personId = null, ids = null) {
  const observationIds = ids || [...state.selected];
  if (!observationIds.length) return;
  state.busy = true;
  render();
  try {
    const body = { observation_ids: observationIds, action };
    if (personId) body.person_id = Number(personId);
    const result = await jsonFetch("/api/faces/review/bulk", {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.message = `${result.changed} observation${result.changed === 1 ? "" : "s"} updated`;
    state.selected.clear();
    await Promise.all([
      loadTab("review"),
      jsonFetch("/api/faces/people/representation-health").then((data) => { state.health = data; }),
    ]);
  } catch (error) {
    state.message = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

async function optimizePerson(personId, apply) {
  state.busy = true;
  render();
  try {
    const result = await jsonFetch(
      `/api/faces/people/${encodeURIComponent(personId)}/gallery/optimize?max_references=8&apply=${apply ? "true" : "false"}`,
      { method: "POST" },
    );
    const baseline = result.baseline?.rank_one_accuracy;
    const optimized = result.optimized?.rank_one_accuracy;
    state.message = result.reason
      ? `${result.name || "Person"}: ${result.reason.replaceAll("_", " ")}`
      : `${result.name}: ${pct(baseline)} → ${pct(optimized)}${result.applied ? " applied" : result.improved ? " available" : " no change"}`;
    state.health = await jsonFetch("/api/faces/people/representation-health");
  } catch (error) {
    state.message = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

async function loadClusterMembers(clusterId) {
  state.busy = true;
  render();
  try {
    const members = await jsonFetch(`/api/faces/unknown-clusters/${encodeURIComponent(clusterId)}/members?limit=100`);
    const cluster = state.clusters.find((item) => Number(item.cluster_id) === Number(clusterId));
    if (cluster) cluster.__members = members;
  } catch (error) {
    state.message = error.message;
  } finally {
    state.busy = false;
    render();
  }
}

function reviewCard(item) {
  const checked = state.selected.has(Number(item.id));
  const reasons = (item.review_reasons || []).map((reason) => badge(reason.replaceAll("_", " "))).join("");
  const candidate = item.candidate_person_id
    ? `<strong>${esc(item.candidate_person_name || `Person ${item.candidate_person_id}`)}</strong> ${score(item.candidate_confidence)}`
    : `<span class="i13-muted">No candidate</span>`;
  return `
    <article class="i13-face-card">
      <label class="i13-check"><input type="checkbox" data-select="${item.id}" ${checked ? "checked" : ""}></label>
      <img src="${cropUrl(item.id)}" loading="lazy" alt="">
      <div class="i13-face-body">
        <div class="i13-face-top">
          <span>${esc(item.camera_id)}</span>
          <strong>${Math.round(Number(item.review_priority || 0) * 100)}</strong>
        </div>
        <div class="i13-candidate">${candidate}</div>
        <small>${esc(item.observed_at || "")}</small>
        <div class="i13-badges">${reasons}</div>
        <div class="i13-metrics">
          <span>quality ${score(item.quality_score)}</span>
          <span>margin ${score(item.match_details?.margin)}</span>
          <span>cluster ${item.cluster_size || 1}</span>
        </div>
        ${item.candidate_person_id ? `<button class="i13-mini primary" data-assign-one="${item.id}" data-person="${item.candidate_person_id}">Assign ${esc(item.candidate_person_name || "")}</button>` : ""}
      </div>
    </article>`;
}

function renderReview() {
  const selectedCount = state.selected.size;
  return `
    <div class="i13-toolbar">
      <strong>${state.review.length} prioritized observations</strong>
      <div class="i13-toolbar-actions">
        <select id="i13-person-select">
          <option value="">Assign selected to…</option>
          ${state.people.map((person) => `<option value="${person.id}" ${String(person.id) === state.selectedPersonId ? "selected" : ""}>${esc(person.name)}</option>`).join("")}
        </select>
        <button data-bulk-assign ${selectedCount ? "" : "disabled"}>Assign ${selectedCount || ""}</button>
        <button data-bulk-unassign ${selectedCount ? "" : "disabled"}>Unassign ${selectedCount || ""}</button>
      </div>
    </div>
    <div class="i13-face-grid">
      ${state.review.map(reviewCard).join("") || `<div class="i13-empty">No observations need review.</div>`}
    </div>`;
}

function renderPeople() {
  const byId = new Map(state.people.map((person) => [Number(person.id), person]));
  return `<div class="i13-people-grid">
    ${state.health.map((item) => {
      const person = byId.get(Number(item.person_id));
      const flags = (item.flags || []).map((flag) => badge(flag.replaceAll("_", " "), flag.includes("weak") || flag.includes("low") ? "warn" : "")).join("");
      return `<article class="i13-person-card">
        <header><div><strong>${esc(item.name)}</strong><small>${item.sample_count} samples · ${item.camera_count} cameras</small></div><span class="i13-refcount">${item.pinned_references} refs</span></header>
        <div class="i13-person-stats">
          <span><small>Median model</small><strong>${score(item.model_score?.median)}</strong></span>
          <span><small>Same-person</small><strong>${score(item.same_person?.median)}</strong></span>
          <span><small>Separation</small><strong>${score(item.separation)}</strong></span>
        </div>
        <div class="i13-badges">${flags || badge("healthy representation", "ok")}</div>
        <div class="i13-person-actions">
          <button data-preview-opt="${item.person_id}">Preview optimize</button>
          <button class="primary" data-apply-opt="${item.person_id}" ${item.sample_count < 2 ? "disabled" : ""}>Optimize gallery</button>
        </div>
      </article>`;
    }).join("")}
  </div>`;
}

function renderConfirmed() {
  return `<div class="i13-confirmed-list">
    ${state.confirmed.map((item) => `
      <article class="i13-confirmed-row">
        <img src="${cropUrl(item.observation_id)}" loading="lazy" alt="">
        <div>
          <strong>${esc(item.name)}</strong>
          <small>${esc(item.camera_id)} · ${esc(item.observed_at)}</small>
          <div class="i13-badges">${(item.flags || []).map((flag) => badge(flag.replaceAll("_", " "), "warn")).join("")}</div>
        </div>
        <div class="i13-confirmed-metrics">
          <span>true <strong>${score(item.true_score)}</strong></span>
          <span>nearest <strong>${esc(item.nearest_competing_name)}</strong> ${score(item.nearest_competing_score)}</span>
          <span>margin <strong>${score(item.margin)}</strong></span>
        </div>
      </article>`).join("") || `<div class="i13-empty">No confirmed diagnostics.</div>`}
  </div>`;
}

function renderClusters() {
  return `<div class="i13-cluster-list">
    ${state.clusters.slice(0, 40).map((cluster) => `
      <article class="i13-cluster-card">
        <header>
          <div><strong>${esc(cluster.name || `Unknown Person ${cluster.cluster_id}`)}</strong><small>${cluster.observation_count || 0} observations · ${cluster.camera_count || 0} cameras</small></div>
          <button data-cluster="${cluster.cluster_id}">${cluster.__members ? "Refresh" : "Inspect"}</button>
        </header>
        ${cluster.__members ? `<div class="i13-cluster-members">${cluster.__members.map((item) => `<img src="${cropUrl(item.id)}" loading="lazy" alt="" title="${esc(item.camera_id)}">`).join("") || "<span>No canonical members.</span>"}</div>` : ""}
      </article>`).join("") || `<div class="i13-empty">No recurring unknown clusters.</div>`}
  </div>`;
}

function render() {
  const mount = document.getElementById("survng-identity-13");
  if (!mount) return;
  const body = state.tab === "review" ? renderReview()
    : state.tab === "people" ? renderPeople()
      : state.tab === "confirmed" ? renderConfirmed()
        : renderClusters();

  mount.innerHTML = `
    <section class="i13-shell">
      <header class="i13-head">
        <div><strong>Identity Operations</strong><small>Review, gallery health, and recurring unknowns</small></div>
        <button data-refresh title="Refresh">↻</button>
      </header>
      <nav class="i13-tabs">
        ${[
          ["review", "Review"],
          ["people", "People"],
          ["confirmed", "Diagnostics"],
          ["clusters", "Unknowns"],
        ].map(([id, label]) => `<button class="${state.tab === id ? "active" : ""}" data-tab="${id}">${label}</button>`).join("")}
      </nav>
      ${state.message ? `<div class="i13-message">${esc(state.message)}</div>` : ""}
      ${state.busy ? `<div class="i13-loading">Working…</div>` : ""}
      <div class="i13-content">${body}</div>
    </section>`;
  wire();
}

function wire() {
  document.querySelectorAll("#survng-identity-13 [data-tab]").forEach((el) => {
    el.onclick = () => loadTab(el.dataset.tab);
  });
  document.querySelectorAll("#survng-identity-13 [data-select]").forEach((el) => {
    el.onchange = () => {
      const id = Number(el.dataset.select);
      if (el.checked) state.selected.add(id); else state.selected.delete(id);
      render();
    };
  });
  const personSelect = document.getElementById("i13-person-select");
  if (personSelect) personSelect.onchange = () => { state.selectedPersonId = personSelect.value; };
  const bulkAssign = document.querySelector("#survng-identity-13 [data-bulk-assign]");
  if (bulkAssign) bulkAssign.onclick = () => {
    if (state.selectedPersonId) bulkAction("assign", state.selectedPersonId);
  };
  const bulkUnassign = document.querySelector("#survng-identity-13 [data-bulk-unassign]");
  if (bulkUnassign) bulkUnassign.onclick = () => bulkAction("unassign");
  document.querySelectorAll("#survng-identity-13 [data-assign-one]").forEach((el) => {
    el.onclick = () => bulkAction("assign", el.dataset.person, [Number(el.dataset.assignOne)]);
  });
  document.querySelectorAll("#survng-identity-13 [data-preview-opt]").forEach((el) => {
    el.onclick = () => optimizePerson(el.dataset.previewOpt, false);
  });
  document.querySelectorAll("#survng-identity-13 [data-apply-opt]").forEach((el) => {
    el.onclick = () => optimizePerson(el.dataset.applyOpt, true);
  });
  document.querySelectorAll("#survng-identity-13 [data-cluster]").forEach((el) => {
    el.onclick = () => loadClusterMembers(el.dataset.cluster);
  });
  const refresh = document.querySelector("#survng-identity-13 [data-refresh]");
  if (refresh) refresh.onclick = () => loadBase();
}

function installStyle() {
  if (document.getElementById("survng-identity-13-style")) return;
  const style = document.createElement("style");
  style.id = "survng-identity-13-style";
  style.textContent = `
#survng-identity-13{max-width:1600px;margin:0 auto 24px;padding:0 18px;font-family:inherit}
.i13-shell{border:1px solid var(--border,#29313d);border-radius:14px;background:var(--panel,#111820);overflow:hidden}
.i13-head{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--border,#29313d)}
.i13-head>div{display:flex;flex-direction:column;gap:2px}.i13-head small,.i13-muted,.i13-person-card small,.i13-confirmed-row small,.i13-cluster-card small{opacity:.68}
.i13-head button,.i13-tabs button,.i13-toolbar button,.i13-person-actions button,.i13-cluster-card button,.i13-mini{border:1px solid var(--border,#3a4553);background:transparent;color:inherit;border-radius:8px;padding:7px 10px;cursor:pointer}
.i13-tabs{display:flex;gap:4px;padding:8px 12px;border-bottom:1px solid var(--border,#29313d);overflow:auto}
.i13-tabs button.active,.i13-mini.primary,.i13-person-actions .primary{background:var(--accent,#2f77d0);color:#fff;border-color:transparent}
.i13-content{padding:12px}.i13-loading,.i13-message{padding:8px 14px}.i13-message{background:rgba(67,137,255,.1)}
.i13-toolbar{display:flex;gap:10px;align-items:center;justify-content:space-between;margin-bottom:10px}.i13-toolbar-actions{display:flex;gap:7px;align-items:center}.i13-toolbar select{max-width:210px}
.i13-face-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}
.i13-face-card{position:relative;display:grid;grid-template-columns:92px 1fr;min-height:128px;border:1px solid var(--border,#29313d);border-radius:12px;overflow:hidden;background:rgba(255,255,255,.025)}
.i13-face-card>img{width:92px;height:100%;object-fit:cover;background:#000}.i13-check{position:absolute;left:6px;top:6px;background:rgba(0,0,0,.6);padding:3px;border-radius:6px}
.i13-face-body{padding:9px;min-width:0}.i13-face-top{display:flex;justify-content:space-between;gap:8px}.i13-candidate{margin-top:4px}.i13-face-body small{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:.55;font-size:10px}
.i13-badges{display:flex;gap:4px;flex-wrap:wrap;margin-top:6px}.i13-badge{font-size:10px;border:1px solid currentColor;border-radius:999px;padding:2px 5px;opacity:.78}.i13-badge.warn{color:#f2ae49}.i13-badge.ok{color:#55bd78}
.i13-metrics{display:flex;gap:8px;flex-wrap:wrap;font-size:10px;opacity:.7;margin:6px 0}.i13-mini{font-size:11px;padding:4px 7px}
.i13-people-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}.i13-person-card{border:1px solid var(--border,#29313d);border-radius:12px;padding:12px}.i13-person-card header{display:flex;justify-content:space-between}.i13-person-card header>div{display:flex;flex-direction:column}.i13-refcount{font-weight:700}
.i13-person-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:12px 0}.i13-person-stats span{display:flex;flex-direction:column;background:rgba(255,255,255,.035);border-radius:8px;padding:7px}.i13-person-stats small{font-size:10px}.i13-person-actions{display:flex;gap:6px;margin-top:10px}
.i13-confirmed-list,.i13-cluster-list{display:flex;flex-direction:column;gap:8px}.i13-confirmed-row{display:grid;grid-template-columns:64px minmax(160px,1fr) minmax(210px,auto);gap:10px;align-items:center;border:1px solid var(--border,#29313d);border-radius:10px;padding:7px}.i13-confirmed-row img{width:64px;height:64px;object-fit:cover;border-radius:8px}.i13-confirmed-row>div{display:flex;flex-direction:column}.i13-confirmed-metrics{font-size:12px}
.i13-cluster-card{border:1px solid var(--border,#29313d);border-radius:10px;padding:10px}.i13-cluster-card header{display:flex;align-items:center;justify-content:space-between}.i13-cluster-card header>div{display:flex;flex-direction:column}.i13-cluster-members{display:flex;gap:5px;overflow:auto;margin-top:8px}.i13-cluster-members img{width:64px;height:64px;object-fit:cover;border-radius:7px}
.i13-empty{padding:24px;text-align:center;opacity:.6}
@media(max-width:720px){#survng-identity-13{padding:0 8px}.i13-toolbar{align-items:stretch;flex-direction:column}.i13-toolbar-actions{display:grid;grid-template-columns:1fr 1fr}.i13-toolbar select{grid-column:1/-1;max-width:none}.i13-face-grid{grid-template-columns:1fr}.i13-confirmed-row{grid-template-columns:56px 1fr}.i13-confirmed-metrics{grid-column:1/-1;display:flex!important;flex-direction:row!important;gap:10px;flex-wrap:wrap}.i13-person-stats{gap:4px}}
  `;
  document.head.appendChild(style);
}

function ensureMounted() {
  const existing = document.getElementById("survng-identity-13");
  if (!isPeoplePath()) {
    existing?.remove();
    return;
  }
  if (existing) return;
  installStyle();
  const mount = document.createElement("div");
  mount.id = "survng-identity-13";
  const main = document.querySelector("main") || document.body;
  main.prepend(mount);
  render();
  loadBase();
}

window.addEventListener("popstate", () => setTimeout(ensureMounted, 0));
document.addEventListener("DOMContentLoaded", ensureMounted);
setInterval(ensureMounted, 1000);
ensureMounted();
