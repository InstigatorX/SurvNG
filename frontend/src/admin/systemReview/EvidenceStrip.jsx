import React from "react";
import { mediaUrl } from "../../shared/api.js";
import { verdictLabel } from "../../systemReviewView.mjs";

export function EvidenceStrip({ samples = [] }) {
  const pictures = samples.filter((sample) => sample?.image_url);
  if (!pictures.length) return null;
  return (
    <div className="system-review-evidence">
      {pictures.map((sample, index) => (
        <figure key={`${sample.record_id || sample.id || sample.image_url}-${index}`}>
          <img src={mediaUrl(sample.image_url)} alt={sample.summary || verdictLabel(sample.verdict)} loading="lazy" />
          {sample.verdict ? <figcaption>{verdictLabel(sample.verdict)}</figcaption> : null}
          {sample.summary ? <figcaption>{sample.summary}</figcaption> : null}
        </figure>
      ))}
    </div>
  );
}
