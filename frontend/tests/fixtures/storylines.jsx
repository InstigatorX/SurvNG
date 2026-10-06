import React from "react";
import { createRoot } from "react-dom/client";
import { StorylinesPage } from "../../src/storylines/StorylinesPage.jsx";
import "../../src/styles.css";
createRoot(document.getElementById("root")).render(<StorylinesPage timeZone="America/New_York" canEdit={!new URLSearchParams(window.location.search).has("viewer")} />);
