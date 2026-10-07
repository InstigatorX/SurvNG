import React from "react";
import { Clapperboard, Clock3, Search } from "lucide-react";
import { appUrl } from "../shared/api.js";
import { workspaceHref } from "../workspaceNavigation.mjs";
import { IncidentsPage } from "../incidents/IncidentsPage.jsx";
import { RecordingsPage, SemanticSearchPage } from "../timeline/TimelinePages.jsx";
import "./review.css";

function reviewEpoch(incident) {
  if (!incident) return null;
  const direct = Number(incident.start_epoch);
  if (Number.isFinite(direct) && direct > 0) return direct;
  const parsed = Date.parse(incident.start_at || incident.created_at || "");
  return Number.isFinite(parsed) ? parsed / 1000 : null;
}

function reviewEventId(incident) {
  const value = Number(incident?.representative_event_id || incident?.event_id);
  return Number.isInteger(value) && value > 0 ? value : null;
}

function ReviewModes({ mode }) {
  const items = [
    ["incidents", "Incidents", workspaceHref("review"), Clapperboard],
    ["timeline", "Timeline", workspaceHref("review", { mode: "timeline" }), Clock3],
    ["search", "Search", workspaceHref("review", { mode: "search" }), Search],
  ];
  return (
    <nav className="review-modes" aria-label="Review mode">
      {items.map(([id, label, href, Icon]) => (
        <a key={id} href={appUrl(href)} className={mode === id ? "active" : ""} aria-current={mode === id ? "page" : undefined}>
          <Icon size={15} />{label}
        </a>
      ))}
    </nav>
  );
}

export function ReviewPage({ mode = "incidents", timeZone, canCorrectIncident = false, onRecordingContextChange, onAssistantContextChange, onAskAssistant = null }) {
  const reviewMode = mode === "timeline" || mode === "search" ? mode : "incidents";
  return (
    <div className={`review-frame review-mode-${reviewMode}`}>
      <ReviewModes mode={reviewMode} />
      {reviewMode === "search" ? (
        <SemanticSearchPage timeZone={timeZone} onAssistantContextChange={onAssistantContextChange} />
      ) : reviewMode === "timeline" ? (
        <RecordingsPage timeZone={timeZone} onAssistantContextChange={onAssistantContextChange} onAskAssistant={onAskAssistant} />
      ) : (
        <IncidentsPage
          desk
          timeZone={timeZone}
          canCorrectIncident={canCorrectIncident}
          onRecordingContextChange={onRecordingContextChange}
          onAssistantContextChange={onAssistantContextChange}
          onAskAssistant={onAskAssistant}
          renderPlayer={(incident) => (
            <RecordingsPage
              embedded
              timeZone={timeZone}
              onAskAssistant={onAskAssistant}
              onAssistantContextChange={onAssistantContextChange}
              controlledCameraId={incident?.camera_id || ""}
              controlledEpoch={reviewEpoch(incident)}
              controlledEventId={reviewEventId(incident)}
            />
          )}
        />
      )}
    </div>
  );
}
