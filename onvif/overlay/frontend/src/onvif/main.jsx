import React from "react";
import { createRoot } from "react-dom/client";
import OnvifInspector from "./OnvifInspector";
import "./onvif.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <OnvifInspector />
  </React.StrictMode>
);
