import { useEffect, useRef } from "react";
import { appUrl } from "./api.js";
import { APP_EVENT_TYPES } from "./constants.js";
import { appEventTypesKey } from "./appEventTypes.mjs";

export { appEventTypesKey };

const appEventListeners = new Set();
// Event types each listener reads; null means every type.
const appEventListenerTypes = new Map();
let appEventSource = null;
let appEventSourceTypes = "";
let appEventReconcileTimer = null;
let appEventCloseTimer = null;
let lastAppEventId = "";
let visibilityListenerAttached = false;

function wantedTypesKey() {
  return appEventTypesKey(appEventListenerTypes.values());
}

function streamUrl(typesKey) {
  const params = new URLSearchParams();
  if (lastAppEventId) params.set("last_event_id", lastAppEventId);
  if (typesKey) params.set("types", typesKey);
  const url = appUrl("/api/events/stream");
  return params.size ? `${url}?${params}` : url;
}

function documentIsHidden() {
  return typeof document !== "undefined" && document.visibilityState === "hidden";
}

function closeAppEventSource() {
  const source = appEventSource;
  appEventSource = null;
  source?.close();
}

function rememberEventId(event) {
  if (event.lastEventId) lastAppEventId = event.lastEventId;
}

function connectAppEventSource() {
  if (!appEventListeners.size) return;
  if (documentIsHidden()) {
    closeAppEventSource();
    return;
  }
  const typesKey = wantedTypesKey();
  if (appEventSource && appEventSourceTypes === typesKey) return;
  // Listeners now need a different set of types: reconnect from the cursor.
  closeAppEventSource();
  const source = new EventSource(streamUrl(typesKey));
  appEventSource = source;
  appEventSourceTypes = typesKey;
  source.addEventListener("connected", (event) => {
    if (source !== appEventSource) return;
    rememberEventId(event);
    // A restarted server or an expired replay cursor cannot replay all incident
    // changes. Readers reconcile their current view after every connection.
    appEventListeners.forEach((current) => current({ type: "resync", data: {}, id: event.lastEventId }));
  });
  APP_EVENT_TYPES.forEach((type) => {
    source.addEventListener(type, (event) => {
      if (source !== appEventSource) return;
      let data = null;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      rememberEventId(event);
      appEventListeners.forEach((current) => current({ type, data, id: event.lastEventId }));
    });
  });
}

function handleVisibilityChange() {
  if (documentIsHidden()) {
    closeAppEventSource();
  } else {
    connectAppEventSource();
  }
}

function attachVisibilityListener() {
  if (visibilityListenerAttached || typeof document === "undefined") return;
  document.addEventListener("visibilitychange", handleVisibilityChange);
  visibilityListenerAttached = true;
}

function detachVisibilityListener() {
  if (!visibilityListenerAttached || typeof document === "undefined") return;
  document.removeEventListener("visibilitychange", handleVisibilityChange);
  visibilityListenerAttached = false;
}

// A page change unsubscribes one page before the next subscribes; settle
// before narrowing the stream so it does not reconnect twice.
function scheduleAppEventConnect() {
  if (appEventReconcileTimer) return;
  appEventReconcileTimer = window.setTimeout(() => {
    appEventReconcileTimer = null;
    connectAppEventSource();
  }, 0);
}

export function subscribeAppEvents(listener, types = null) {
  appEventListeners.add(listener);
  appEventListenerTypes.set(listener, Array.isArray(types) && types.length ? [...types] : null);
  attachVisibilityListener();
  if (appEventCloseTimer) {
    window.clearTimeout(appEventCloseTimer);
    appEventCloseTimer = null;
  }
  connectAppEventSource();
  return () => {
    appEventListeners.delete(listener);
    appEventListenerTypes.delete(listener);
    if (appEventListeners.size) scheduleAppEventConnect();
    if (!appEventListeners.size) {
      detachVisibilityListener();
      if (!appEventSource) return;
      appEventCloseTimer = window.setTimeout(() => {
        if (!appEventListeners.size && appEventSource) {
          closeAppEventSource();
        }
        appEventCloseTimer = null;
      }, 1000);
    }
  };
}

/** `types`: the event types this handler reads; omit for every type. */
export function useAppEvents(handler, enabled = true, types = null) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;
  const typesKey = Array.isArray(types) ? types.join(",") : "";
  useEffect(() => (
    enabled ? subscribeAppEvents((event) => handlerRef.current(event), typesKey ? typesKey.split(",") : null) : undefined
  ), [enabled, typesKey]);
}
