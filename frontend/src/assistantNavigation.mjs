function internalPath(value) {
  const path = String(value || "").trim();
  return path.startsWith("/") && !path.startsWith("//") ? path : "";
}

export function assistantIncidentHref(eventId) {
  const normalized = Number(eventId);
  if (!Number.isInteger(normalized) || normalized <= 0) return "";
  return `/incidents?event_ids=${normalized}`;
}

export function assistantEvidenceHref(item) {
  const href = internalPath(item?.href);
  if (href) return href;
  const imageUrl = internalPath(item?.image_url);
  const eventMatch = imageUrl.match(/^\/api\/events\/(\d+)\//);
  return eventMatch ? assistantIncidentHref(eventMatch[1]) : imageUrl;
}
