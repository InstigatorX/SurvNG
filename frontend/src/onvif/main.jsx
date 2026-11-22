import React from "react";
import { createRoot } from "react-dom/client";
import OnvifInspector from "./OnvifInspector";
import "../typeScale.css";
import "./onvif.css";
import { applyTextSize, TEXT_SIZE_STORAGE_KEY } from "../textSize.mjs";

// Honor the per-browser text size chosen in Admin > Server.
try {
  applyTextSize(window.localStorage.getItem(TEXT_SIZE_STORAGE_KEY));
} catch {
  applyTextSize("default");
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <OnvifInspector />
  </React.StrictMode>
);
