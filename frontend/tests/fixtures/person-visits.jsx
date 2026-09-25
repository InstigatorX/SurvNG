import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { VisitsPanel } from "../../src/people/VisitsPanel.jsx";
import "../../src/styles.css";
import "../../src/shell/shell.css";
import "../../src/search/search.css";
import "../../src/people/people.css";
import "../../src/shell/responsive.css";
import "../../src/shell/mobile.css";

function Fixture() {
  const [personId, setPersonId] = useState("");
  return <main className="faces-page faces-mode-visits">
    <header className="faces-commandbar"><label>Person filter<select value={personId} onChange={(event) => setPersonId(event.target.value)}>
      <option value="">All people</option><option value="7">Alex</option><option value="8">Sam</option>
    </select></label></header>
    <nav className="people-mode-tabs" aria-label="People workspace mode">Visits</nav>
    <aside className="faces-people-panel">People</aside>
    <section className="faces-review-panel">
      <VisitsPanel people={[{ id: 7, name: "Alex" }, { id: 8, name: "Sam" }]} personId={personId} timeZone="UTC" />
    </section>
  </main>;
}
createRoot(document.getElementById("root")).render(<Fixture />);
