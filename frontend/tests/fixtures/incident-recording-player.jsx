import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { IncidentRecordingPlayer } from "../../src/incidents/IncidentRecordingPlayer.jsx";

function Fixture() {
  const params = new URLSearchParams(location.search);
  const [ended, setEnded] = useState(0);
  return <>
    <IncidentRecordingPlayer cameraId="fixture" startEpoch={Number(params.get("start") || 900)}
      endEpoch={Number(params.get("end") || 906)} autoPlay={params.get("autoplay") !== "false"}
      onEnded={() => setEnded((count) => count + 1)} />
    <output id="ended">{ended}</output>
  </>;
}

createRoot(document.getElementById("root")).render(<Fixture />);
