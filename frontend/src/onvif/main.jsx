import React from "react";
import { createRoot } from "react-dom/client";
import OnvifInspector from "./OnvifInspector";
import "../typeScale.css";
import "../fonts.css";
import "./onvif.css";
import { applyTextSize, TEXT_SIZE_STORAGE_KEY } from "../textSize.mjs";
import { applyInterfaceFont, INTERFACE_FONT_STORAGE_KEY } from "../interfaceFont.mjs";

// Honor the per-browser text size and font chosen in Admin > General.
try {
  applyTextSize(window.localStorage.getItem(TEXT_SIZE_STORAGE_KEY));
  applyInterfaceFont(window.localStorage.getItem(INTERFACE_FONT_STORAGE_KEY));
} catch {
  applyTextSize("default");
  applyInterfaceFont("inter");
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <OnvifInspector />
  </React.StrictMode>
);
