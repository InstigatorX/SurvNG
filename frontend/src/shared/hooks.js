import { useEffect, useRef, useState } from "react";
import { browserStorage, readStoredValue, writeStoredValue } from "../storage.mjs";
import { preferredStoredValue } from "../adminWorkspace.mjs";

export function useStoredState(key, initialValue, { preferInitial = false } = {}) {
  const [value, setValue] = useState(() => preferredStoredValue(initialValue, readStoredValue(browserStorage(window), key, initialValue), preferInitial));
  useEffect(() => {
    writeStoredValue(browserStorage(window), key, value);
  }, [key, value]);
  return [value, setValue];
}

export function useStoredJsonState(key, initialValue) {
  const [value, setValue] = useState(() => {
    const stored = readStoredValue(browserStorage(window), key, "");
    if (!stored) return initialValue;
    try { return JSON.parse(stored); } catch { return initialValue; }
  });
  useEffect(() => {
    writeStoredValue(browserStorage(window), key, JSON.stringify(value));
  }, [key, value]);
  return [value, setValue];
}
export function isMobileViewport() {
  return typeof window !== "undefined" && window.matchMedia("(max-width: 760px)").matches;
}

export function useViewportQuery(query) {
  const [matches, setMatches] = useState(() => typeof window !== "undefined" && window.matchMedia(query).matches);
  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    media.addEventListener?.("change", update);
    return () => media.removeEventListener?.("change", update);
  }, [query]);
  return matches;
}

export function useModalFocus(onClose) {
  const modalRef = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const modal = modalRef.current;
    const root = document.getElementById("root");
    const returnTarget = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    const rootWasInert = root?.inert || false;
    const rootAriaHidden = root?.getAttribute("aria-hidden");
    document.body.style.overflow = "hidden";
    if (root) {
      root.inert = true;
      root.setAttribute("aria-hidden", "true");
    }
    window.requestAnimationFrame(() => (modal?.querySelector("[data-modal-initial]") || modal?.querySelector("button, [href], input, select, textarea"))?.focus());
    function onKeyDown(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current?.();
        return;
      }
      if (event.key !== "Tab" || !modal) return;
      const controls = [...modal.querySelectorAll('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])')];
      if (!controls.length) return;
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
      if (root) {
        root.inert = rootWasInert;
        if (rootAriaHidden == null) root.removeAttribute("aria-hidden");
        else root.setAttribute("aria-hidden", rootAriaHidden);
      }
      window.requestAnimationFrame(() => {
        if (returnTarget instanceof HTMLElement && returnTarget.isConnected) returnTarget.focus();
      });
    };
  }, []);
  return modalRef;
}
