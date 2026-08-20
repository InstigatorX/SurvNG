import React, { createContext, useContext, useMemo, useRef, useState } from "react";
import { fetch } from "./api.js";
import { useAppEvents } from "./events.js";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { loadRuntimeState } from "./runtimeState.mjs";

const RuntimeStateContext = createContext(null);

export function RuntimeStateProvider({ children }) {
  const [cameras, setCameras] = useState([]);
  const [appConfig, setAppConfig] = useState(null);
  const [system, setSystem] = useState(null);
  const [loading, setLoading] = useState(true);
  const sequenceRef = useRef(0);

  async function refresh(signal) {
    const sequence = ++sequenceRef.current;
    try {
      const { cameras: cameraPayload, appConfig: configPayload, system: systemPayload } = await loadRuntimeState(fetch, { signal });
      if (sequence !== sequenceRef.current) return;
      if (cameraPayload) setCameras(cameraPayload);
      if (configPayload) setAppConfig(configPayload);
      if (systemPayload) setSystem(systemPayload);
    } finally {
      if (sequence === sequenceRef.current) setLoading(false);
    }
  }

  useAppEvents(({ type, data }) => {
    if (type === "cameras_state" && Array.isArray(data)) setCameras(data);
    else if (type === "camera_state" && data?.id) {
      setCameras((current) => {
        const index = current.findIndex((camera) => camera.id === data.id);
        if (index < 0) return [...current, data];
        const next = [...current];
        next[index] = data;
        return next;
      });
    } else if (type === "system_state" && data) setSystem(data);
  });

  useVisiblePolling(refresh, 60_000);
  const value = useMemo(() => ({ cameras, appConfig, system, loading, refresh }), [appConfig, cameras, loading, system]);
  return <RuntimeStateContext.Provider value={value}>{children}</RuntimeStateContext.Provider>;
}

export function useRuntimeState() {
  return useContext(RuntimeStateContext);
}
