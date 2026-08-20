import assert from "node:assert/strict";

const documentListeners = new Map();
const pendingTimers = new Map();
let nextTimer = 1;

Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { userAgent: "node", platform: "Linux", maxTouchPoints: 0 },
});
globalThis.document = {
  visibilityState: "visible",
  documentElement: { dataset: {} },
  addEventListener(type, listener) {
    documentListeners.set(type, listener);
  },
  removeEventListener(type, listener) {
    if (documentListeners.get(type) === listener) documentListeners.delete(type);
  },
};
globalThis.window = {
  __SURVNG_BASE_PATH__: "/survng",
  location: { origin: "http://survng.test", pathname: "/survng/" },
  setTimeout(callback) {
    const id = nextTimer;
    nextTimer += 1;
    pendingTimers.set(id, callback);
    return id;
  },
  clearTimeout(id) {
    pendingTimers.delete(id);
  },
};
window.self = window;
window.top = window;

class FakeEventSource {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.closed = false;
    this.listeners = new Map();
    FakeEventSource.instances.push(this);
  }

  addEventListener(type, listener) {
    this.listeners.set(type, listener);
  }

  close() {
    this.closed = true;
  }

  emit(type, data, lastEventId = "") {
    this.listeners.get(type)?.({ data: JSON.stringify(data), lastEventId });
  }
}

globalThis.EventSource = FakeEventSource;

const { subscribeAppEvents } = await import("../src/shared/events.js");
const received = [];
const unsubscribe = subscribeAppEvents((event) => received.push(event));

assert.equal(FakeEventSource.instances.length, 1);
assert.equal(FakeEventSource.instances[0].url, "/survng/api/events/stream");
assert.ok(documentListeners.has("visibilitychange"));

const firstSource = FakeEventSource.instances[0];
firstSource.emit("connected", { instance: "abc" }, "abc:7");
document.visibilityState = "hidden";
documentListeners.get("visibilitychange")();
assert.equal(firstSource.closed, true);

document.visibilityState = "visible";
documentListeners.get("visibilitychange")();
assert.equal(FakeEventSource.instances.length, 2);
assert.equal(
  FakeEventSource.instances[1].url,
  "/survng/api/events/stream?last_event_id=abc%3A7",
);

const resumedSource = FakeEventSource.instances[1];
resumedSource.emit("camera_state", { id: "gate", running: true }, "abc:8");
assert.deepEqual(received, [{
  type: "camera_state",
  data: { id: "gate", running: true },
  id: "abc:8",
}]);
resumedSource.emit("identity_update", { event_id: 42, person_id: 7 }, "abc:9");
assert.deepEqual(received.at(-1), {
  type: "identity_update",
  data: { event_id: 42, person_id: 7 },
  id: "abc:9",
});

unsubscribe();
assert.equal(documentListeners.has("visibilitychange"), false);
assert.equal(pendingTimers.size, 1);

document.visibilityState = "hidden";
const hiddenUnsubscribe = subscribeAppEvents(() => {});
assert.equal(pendingTimers.size, 0);
assert.equal(resumedSource.closed, true);
assert.equal(FakeEventSource.instances.length, 2);
document.visibilityState = "visible";
documentListeners.get("visibilitychange")();
assert.equal(FakeEventSource.instances.length, 3);
assert.equal(
  FakeEventSource.instances[2].url,
  "/survng/api/events/stream?last_event_id=abc%3A9",
);
hiddenUnsubscribe();
assert.equal(pendingTimers.size, 1);
pendingTimers.values().next().value();
assert.equal(FakeEventSource.instances[2].closed, true);

console.log("application event stream tests passed");
