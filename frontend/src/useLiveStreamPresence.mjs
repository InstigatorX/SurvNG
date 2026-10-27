import { useEffect, useState } from "react";
import { nextLivePresence } from "./liveStreamPresence.mjs";

const ACTIVITY_EVENTS = ["pointerdown", "keydown", "wheel", "touchstart"];
const POINTER_MOVE_INTERVAL_MS = 1000;

const listeners = new Set();
let state = {
  active: true,
  documentVisible: true,
  lastActivityAt: 0,
};
let started = false;
let idleTimer = 0;
let lastPointerMoveAt = 0;

function emit() {
  const snapshot = { active: state.active, documentVisible: state.documentVisible };
  listeners.forEach((listener) => listener(snapshot));
}

function evaluate(lastActivityAt) {
  const now = Date.now();
  const documentVisible = typeof document === "undefined" || !document.hidden;
  const next = nextLivePresence({ lastActivityAt, now, documentVisible });
  const changed = next.active !== state.active || next.documentVisible !== state.documentVisible;
  state = {
    active: next.active,
    documentVisible: next.documentVisible,
    lastActivityAt,
  };
  window.clearTimeout(idleTimer);
  idleTimer = 0;
  if (next.nextCheckMs != null) {
    idleTimer = window.setTimeout(() => evaluate(state.lastActivityAt), next.nextCheckMs);
  }
  if (changed) emit();
}

function noteActivityFromUser() {
  if (typeof document !== "undefined" && document.hidden) {
    evaluate(state.lastActivityAt);
    return;
  }
  evaluate(Date.now());
}

function onPointerMove(event) {
  if (event && event.isTrusted === false) return;
  const now = Date.now();
  if (now - lastPointerMoveAt < POINTER_MOVE_INTERVAL_MS) return;
  lastPointerMoveAt = now;
  noteActivityFromUser();
}

function onActivity(event) {
  if (event && event.isTrusted === false) return;
  noteActivityFromUser();
}

function onVisibility() {
  if (typeof document !== "undefined" && document.hidden) {
    evaluate(state.lastActivityAt);
    return;
  }
  noteActivityFromUser();
}

function start() {
  if (started || typeof window === "undefined" || typeof document === "undefined") return;
  started = true;
  lastPointerMoveAt = 0;
  ACTIVITY_EVENTS.forEach((type) => window.addEventListener(type, onActivity, true));
  window.addEventListener("pointermove", onPointerMove, true);
  document.addEventListener("visibilitychange", onVisibility);
  evaluate(Date.now());
}

function stop() {
  if (!started || typeof window === "undefined") return;
  started = false;
  window.clearTimeout(idleTimer);
  idleTimer = 0;
  ACTIVITY_EVENTS.forEach((type) => window.removeEventListener(type, onActivity, true));
  window.removeEventListener("pointermove", onPointerMove, true);
  document.removeEventListener("visibilitychange", onVisibility);
}

export function noteLiveStreamActivity() {
  if (!started) start();
  noteActivityFromUser();
}

export function readLiveStreamPresence() {
  if (!started && typeof document !== "undefined") {
    return { active: !document.hidden, documentVisible: !document.hidden };
  }
  return { active: state.active, documentVisible: state.documentVisible };
}

export function subscribeLiveStreamPresence(listener) {
  start();
  listeners.add(listener);
  listener(readLiveStreamPresence());
  return () => {
    listeners.delete(listener);
    if (!listeners.size) stop();
  };
}

export function useLiveStreamPresence() {
  const [presence, setPresence] = useState(readLiveStreamPresence);
  useEffect(() => subscribeLiveStreamPresence(setPresence), []);
  return presence;
}
