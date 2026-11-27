import { useEffect, useState } from "react";
import { fetch } from "../shared/api.js";

// Saved stills can render directly. Recording frames are fetched with their
// timestamp header so the player and Timeline link describe what was decoded.
export function useIncidentStill(candidates) {
  const key = JSON.stringify(candidates.map(({ src }) => src));
  const [state, setState] = useState({ key, index: 0, preview: null });
  const current = state.key === key ? state : { key, index: 0, preview: null };
  if (current !== state) setState(current);
  const { index, preview } = current;
  const candidate = candidates[index];
  const previewUrl = candidate?.source ? candidate.src : null;

  useEffect(() => {
    if (!previewUrl) return undefined;
    const controller = new AbortController();
    let objectUrl;
    async function load() {
      try {
        const response = await fetch(previewUrl, { signal: controller.signal });
        if (!response.ok) throw new Error("Recording frame unavailable");
        const blob = await response.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        const image = new Image();
        image.src = objectUrl;
        await image.decode();
        if (controller.signal.aborted) return;
        const actual = Number(response.headers.get("X-SurvNG-Actual-Timestamp"));
        const preview = { src: objectUrl, width: image.naturalWidth, height: image.naturalHeight, actual: Number.isFinite(actual) && actual > 0 ? actual : null };
        setState((current) => current.key === key && current.index === index ? { ...current, preview } : current);
      } catch {
        if (!controller.signal.aborted) {
          setState((current) => current.key === key && current.index === index ? { key, index: index + 1, preview: null } : current);
        }
      }
    }
    void load();
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [key, index, previewUrl]);

  if (!candidate) return { status: "unavailable" };
  if (previewUrl) {
    if (!preview) return { status: "loading" };
    return { ...candidate, ...preview, status: "ready", epoch: preview.actual ?? candidate.epoch };
  }
  return { ...candidate, status: "ready", onError: () => setState((current) => current.key === key && current.index === index
    ? { key, index: index + 1, preview: null } : current) };
}
