import React from "react";
import { createRoot } from "react-dom/client";
import { VisitsPanel } from "../../src/people/VisitsPanel.jsx";
createRoot(document.getElementById("root")).render(<VisitsPanel people={[{ id: 7, name: "Alex" }]} timeZone="UTC" />);
