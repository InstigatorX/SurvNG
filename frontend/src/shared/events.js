import { useEffect, useRef } from "react";
import { appUrl } from "./api.js";
import { APP_EVENT_TYPES } from "./constants.js";

const appEventListeners = new Set();
let appEventSource = null;
let appEventCloseTimer = null;

export function subscribeAppEvents(listener) {
  appEventListeners.add(listener);
  if (appEventCloseTimer) {
    window.clearTimeout(appEventCloseTimer);
    appEventCloseTimer = null;
  }
  if (!appEventSource) {
    appEventSource = new EventSource(appUrl("/api/events/stream"));
    APP_EVENT_TYPES.forEach((type) => {
      appEventSource.addEventListener(type, (event) => {
        let data = null;
        try {
          data = JSON.parse(event.data);
        } catch {
          return;
        }
        appEventListeners.forEach((current) => current({ type, data, id: event.lastEventId }));
      });
    });
  }
  return () => {
    appEventListeners.delete(listener);
    if (!appEventListeners.size && appEventSource) {
      appEventCloseTimer = window.setTimeout(() => {
        if (!appEventListeners.size && appEventSource) {
          appEventSource.close();
          appEventSource = null;
        }
      }, 1000);
    }
  };
}

export function useAppEvents(handler) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;
  useEffect(() => subscribeAppEvents((event) => handlerRef.current(event)), []);
}
