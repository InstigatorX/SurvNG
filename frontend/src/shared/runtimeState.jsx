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
  const [camerasUpdatedAt, setCamerasUpdatedAt] = useState(null);
  const [systemUpdatedAt, setSystemUpdatedAt] = useState(null);
  const [camerasError, setCamerasError] = useState(false);
  const [systemError, setSystemError] = useState(false);
  const [loading, setLoading] = useState(true);
  const sequenceRef = useRef(0);
  const cameraEventSequenceRef = useRef(0);
  const systemEventSequenceRef = useRef(0);

  async function refresh(signal) {
    const sequence = ++sequenceRef.current;
    const cameraEventSequence = cameraEventSequenceRef.current;
    const systemEventSequence = systemEventSequenceRef.current;
    try {
      const { cameras: cameraPayload, appConfig: configPayload, system: systemPayload } = await loadRuntimeState(fetch, { signal });
      if (signal?.aborted || sequence !== sequenceRef.current) return;
      // Events may be newer than this request. Preserve those snapshots without
      // dropping unrelated configuration or system updates from the same poll.
      if (cameraEventSequence === cameraEventSequenceRef.current) {
        if (cameraPayload) {
          setCameras(cameraPayload);
          setCamerasUpdatedAt(Date.now());
          setCamerasError(false);
        } else setCamerasError(true);
      }
      if (configPayload) setAppConfig(configPayload);
      if (systemEventSequence === systemEventSequenceRef.current) {
        if (systemPayload) {
          setSystem(systemPayload);
          setSystemUpdatedAt(Date.now());
          setSystemError(false);
        } else setSystemError(true);
      }
    } finally {
      if (!signal?.aborted && sequence === sequenceRef.current) setLoading(false);
    }
  }

  useAppEvents(({ type, data }) => {
    if (type === "cameras_state" && Array.isArray(data)) {
      cameraEventSequenceRef.current += 1;
      setCameras(data);
      setCamerasUpdatedAt(Date.now());
      setCamerasError(false);
    }
    else if (type === "camera_state" && data?.id) {
      cameraEventSequenceRef.current += 1;
      setCameras((current) => {
        const index = current.findIndex((camera) => camera.id === data.id);
        if (index < 0) return [...current, data];
        const next = [...current];
        next[index] = data;
        return next;
      });
    } else if (type === "system_state" && data) {
      systemEventSequenceRef.current += 1;
      setSystem(data);
      setSystemUpdatedAt(Date.now());
      setSystemError(false);
    }
  });

  useVisiblePolling(refresh, 60_000);
  const value = useMemo(() => ({
    cameras, appConfig, system, loading, refresh,
    camerasUpdatedAt, systemUpdatedAt, camerasError, systemError,
  }), [appConfig, cameras, loading, system, camerasUpdatedAt, systemUpdatedAt, camerasError, systemError]);
  return <RuntimeStateContext.Provider value={value}>{children}</RuntimeStateContext.Provider>;
}

export function useRuntimeState() {
  return useContext(RuntimeStateContext);
}
