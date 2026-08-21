export function registerSurvngServiceWorker() {
  if (typeof window === "undefined" || !("serviceWorker" in navigator)) return;
  if (import.meta.env?.DEV) return;

  const base = String(window.__SURVNG_BASE_PATH__ || "").replace(/\/+$/, "");
  const swUrl = `${base}/sw.js`;
  const scope = base ? `${base}/` : "/";

  const register = () => {
    navigator.serviceWorker.register(swUrl, { scope }).catch(() => {
      // Installability is best-effort; live monitoring still works without a worker.
    });
  };

  if (document.readyState === "complete") register();
  else window.addEventListener("load", register, { once: true });
}
