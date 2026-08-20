import { useEffect, useRef } from "react";
import { appUrl } from "./api.js";
import { APP_EVENT_TYPES } from "./constants.js";

const appEventListeners = new Set();
let appEventSource = null;
let appEventCloseTimer = null;
let lastAppEventId = "";
let visibilityListenerAttached = false;

function streamUrl() {
  const url = appUrl("/api/events/stream");
  if (!lastAppEventId) return url;
  return `${url}?last_event_id=${encodeURIComponent(lastAppEventId)}`;
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
  if (appEventSource || !appEventListeners.size || documentIsHidden()) return;
  const source = new EventSource(streamUrl());
  appEventSource = source;
  source.addEventListener("connected", (event) => {
    if (source === appEventSource) rememberEventId(event);
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

export function subscribeAppEvents(listener) {
  appEventListeners.add(listener);
  attachVisibilityListener();
  if (appEventCloseTimer) {
    window.clearTimeout(appEventCloseTimer);
    appEventCloseTimer = null;
  }
  connectAppEventSource();
  return () => {
    appEventListeners.delete(listener);
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

export function useAppEvents(handler, enabled = true) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;
  useEffect(() => (
    enabled ? subscribeAppEvents((event) => handlerRef.current(event)) : undefined
  ), [enabled]);
}
