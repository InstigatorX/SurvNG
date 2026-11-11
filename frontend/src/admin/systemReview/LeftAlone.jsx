import React from "react";
import { plainNote } from "../../systemReviewView.mjs";

export function LeftAlone({ notes = [], excluded = [], advisories = [], guide = {} }) {
  const skipped = (excluded || []).map((camera) => camera.camera_name || camera.camera_id).filter(Boolean);
  const lines = [
    ...(notes || [])
      .map((note) => plainNote(note?.text, guide))
      .filter((text) => text && !(skipped.length && text.includes("left them out of sampling"))),
    skipped.length
      ? `${skipped.join(", ")} ${skipped.length === 1 ? "was" : "were"} left out of the pictures because detection is off. A site-wide change still applies there if that camera follows the site setting.`
      : "",
    ...(advisories || []),
  ].filter(Boolean);
  if (!lines.length) return null;
  return (
    <details className="system-review-aside">
      <summary>Left alone, and why ({lines.length})</summary>
      <ul>{lines.map((line) => <li key={line}>{line}</li>)}</ul>
    </details>
  );
}
