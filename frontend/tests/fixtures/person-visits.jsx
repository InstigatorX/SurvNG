import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { VisitsPanel } from "../../src/people/VisitsPanel.jsx";
import "../../src/people/people.css";

function Fixture() {
  const [personId, setPersonId] = useState("");
  return <>
    <label>Person filter<select value={personId} onChange={(event) => setPersonId(event.target.value)}>
      <option value="">All people</option><option value="7">Alex</option><option value="8">Sam</option>
    </select></label>
    <VisitsPanel people={[{ id: 7, name: "Alex" }, { id: 8, name: "Sam" }]} personId={personId} timeZone="UTC" />
  </>;
}
createRoot(document.getElementById("root")).render(<Fixture />);
