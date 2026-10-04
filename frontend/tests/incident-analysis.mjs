import assert from "node:assert/strict";
import { analysisLabel, createAnalysisRequester, createIncidentAnalysisViewer } from "../src/incidentAnalysis.mjs";

function harness(initial = "deferred") {
  let time = 100000, shown = true, status = initial, posts = 0, gets = 0, failure = false;
  const changes = [], progress = [];
  const data = () => ({ mode: "on_demand", enabled: true, status, episodes: [], remaining: 1 });
  const fetch = async (_url, options) => {
    if (options?.method === "POST") { posts++; if (failure) throw new Error("offline"); status = "queued"; }
    else gets++;
    return { ok: true, json: async () => data() };
  };
  const request = createAnalysisRequester(fetch, () => time);
  const makeViewer = (extra = {}) => createIncidentAnalysisViewer({ incidentId: "scene/one", fetch, request, visible: () => shown, now: () => time, onChange: (value) => changes.push(value), onProgress: (value) => progress.push(value), ...extra });
  return { makeViewer, changes, progress, get posts() { return posts; }, get gets() { return gets; }, advance(ms) { time += ms; }, hide() { shown = false; }, show() { shown = true; }, status(value) { status = value; }, fail(value) { failure = value; } };
}

{
  const h = harness(), viewer = h.makeViewer();
  await viewer.poll();
  assert.equal(h.posts, 1);
  assert.equal(h.changes.at(-1).status, "queued");
  h.advance(5000); await viewer.poll();
  assert.equal(h.posts, 1, "status polling does not continually renew");
  h.advance(25000); await viewer.poll();
  assert.equal(h.posts, 2, "visible pending work renews at 30 seconds");
  h.hide(); h.advance(60000); await viewer.poll({ force: true });
  assert.equal(h.posts, 2, "hidden tabs do not extend demand");
  h.show(); await viewer.poll({ force: true });
  assert.equal(h.posts, 3);
  viewer.dispose(); h.advance(60000); await viewer.poll({ force: true });
  assert.equal(h.posts, 3, "closed viewers stop work");
}
for (const status of ["complete", "partial", "unavailable"]) {
  const h = harness(status), viewer = h.makeViewer();
  await viewer.poll(); h.advance(60000); await viewer.poll();
  assert.equal(h.posts, 0, `${status} never automatically reruns`);
  assert.equal(h.gets, 1, `${status} stops polling`);
}
{
  const h = harness(), first = h.makeViewer(), second = h.makeViewer();
  await Promise.all([first.poll(), second.poll()]);
  assert.equal(h.posts, 1, "concurrent views share admission");
  first.dispose(); second.dispose();
  const remount = h.makeViewer(); await remount.poll();
  assert.equal(h.posts, 1, "quick remount reuses admission");
}
{
  const h = harness(); h.fail(true);
  const viewer = h.makeViewer(); await viewer.poll();
  assert.match(h.changes.at(-1).error, /Video and saved evidence/);
  h.advance(30000); await viewer.poll();
  assert.equal(h.posts, 1, "failed admission is not repeatedly retried");
  h.fail(false); await viewer.retry();
  assert.equal(h.posts, 2, "explicit retry can admit work");
}
{
  const h = harness();
  let resolve;
  const fetch = () => new Promise((done) => { resolve = done; });
  const viewer = h.makeViewer({ fetch });
  const pending = viewer.poll();
  viewer.dispose();
  resolve({ ok: true, json: async () => ({ enabled: true, status: "deferred" }) });
  await pending;
  assert.equal(h.posts, 0, "unmounted status requests cannot admit work");
  assert.equal(h.changes.length, 0);
}
{
  const h = harness();
  let resolve;
  const viewer = h.makeViewer({ fetch: () => new Promise((done) => { resolve = done; }) });
  const pending = viewer.poll();
  h.hide();
  resolve({ ok: true, json: async () => ({ enabled: true, status: "deferred" }) });
  await pending;
  assert.equal(h.posts, 0, "tab hidden during status request cannot admit work");
}
{
  const h = harness();
  const viewer = h.makeViewer({ fetch: async () => ({ ok: true, json: async () => ({ mode: "eager", enabled: false, status: "complete" }) }) });
  await viewer.poll();
  assert.equal(h.posts, 0, "default eager mode never requests demand");
}
assert.equal(analysisLabel("partial"), "Extra details partly available");
{
  const h = harness(), viewer = h.makeViewer({ autoStart: false });
  await viewer.poll(); h.advance(30000); await viewer.poll();
  assert.equal(h.posts, 0, "desktop feed preview is read-only");
  await viewer.start();
  assert.equal(h.posts, 1, "explicit analysis control admits work");
  h.advance(30000); await viewer.poll();
  assert.equal(h.posts, 2, "explicitly started viewer renews while visible");
}
console.log("Incident analysis admission, deduplication, visibility, renewal, terminal state, error and teardown tests passed");
