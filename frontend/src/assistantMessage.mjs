const CITATION_PATTERN = /\[(E[A-Za-z0-9_-]+)\]/g;

/** Remove grounding markers like [E-system] from reader-facing answer text. */
export function stripAssistantCitationMarkers(text) {
  return String(text || "")
    .replace(/\s*\[(E[A-Za-z0-9_-]+)\]/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .trim();
}

/** Split assistant prose into plain text and citation tokens for interactive rendering. */
export function splitAssistantCitations(text) {
  const source = String(text || "");
  if (!source) return [];
  const parts = [];
  let lastIndex = 0;
  for (const match of source.matchAll(CITATION_PATTERN)) {
    const index = match.index ?? 0;
    if (index > lastIndex) parts.push({ type: "text", value: source.slice(lastIndex, index) });
    parts.push({ type: "citation", value: match[0], evidenceId: match[1] });
    lastIndex = index + match[0].length;
  }
  if (lastIndex < source.length) parts.push({ type: "text", value: source.slice(lastIndex) });
  return parts.length ? parts : [{ type: "text", value: source }];
}

export function assistantEvidenceLabel(evidence) {
  if (!evidence) return "Evidence";
  const title = String(evidence.title || "").trim();
  if (title) return title;
  return String(evidence.id || "Evidence");
}

/** Context-aware status lines while waiting for a full chat response. */
export function assistantThinkingStages(contextLabel = "") {
  const label = String(contextLabel || "").trim();
  const where = label ? label.split(" · ")[1] || label.split(" · ")[0] : "";
  const looking = where ? `Looking at ${where}…` : "Looking at your current view…";
  return [
    looking,
    "Checking what SurvNG recorded…",
    "Putting the answer together…",
  ];
}

export function assistantWelcomeCopy(contextLabel = "") {
  const label = String(contextLabel || "").trim();
  if (!label || label === "Live") {
    return {
      title: "What should we look into?",
      body: "Ask about camera health, recent activity, or a selected incident. I’ll use SurvNG evidence—not guesses.",
    };
  }
  return {
    title: "What should we look into?",
    body: `You’re on ${label}. Ask a question, or pick a suggestion below.`,
  };
}

export function assistantComposerPlaceholder(contextLabel = "") {
  const label = String(contextLabel || "").trim();
  if (!label) return "Ask about your cameras or incidents…";
  const cameraOrPage = label.split(" · ").slice(0, 2).join(" · ");
  return `Ask about ${cameraOrPage}…`;
}

export function assistantCoachSeen(storage, key = "survng.assistantCoach.v1") {
  try {
    return storage?.getItem(key) === "1";
  } catch {
    return false;
  }
}

export function markAssistantCoachSeen(storage, key = "survng.assistantCoach.v1") {
  try {
    storage?.setItem(key, "1");
    return true;
  } catch {
    return false;
  }
}
