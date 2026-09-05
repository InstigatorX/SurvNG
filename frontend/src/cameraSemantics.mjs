const CAMERA_REPORT_CATEGORIES = new Set(["person", "vehicle", "animal", "face"]);

function text(value) {
  return typeof value === "string" ? value.trim() : "";
}

function reportSource(event, report) {
  const eventId = Number(report?.source_event_id ?? report?.event_id ?? event?.representative_event_id ?? event?.id);
  return {
    eventId: Number.isInteger(eventId) && eventId > 0 ? eventId : null,
    eventAt: text(report?.source_created_at) || text(report?.event_at) || text(event?.created_at),
  };
}

export function cameraSemanticsForEvent(event) {
  const direct = event?.camera_semantics;
  const metadata = Array.isArray(event?.objects)
    ? event.objects.find((item) => item?.status === "motion_qualification" && item?.motion_qualification?.camera_semantics)
    : null;
  const semantics = direct || event?.motion_qualification?.camera_semantics || metadata?.motion_qualification?.camera_semantics;
  return semantics && typeof semantics === "object" ? semantics : null;
}

export function cameraReportsForIncident(incident) {
  const events = Array.isArray(incident?.events) && incident.events.length ? incident.events : [incident];
  const reports = [];
  const seen = new Set();
  for (const event of events) {
    const sourceReports = cameraSemanticsForEvent(event)?.reports;
    if (!Array.isArray(sourceReports)) continue;
    for (const report of sourceReports) {
      const topic = text(report?.topic);
      const category = text(report?.category).toLowerCase();
      if (!topic || !CAMERA_REPORT_CATEGORIES.has(category)) continue;
      const reportedClass = text(report?.reported_class);
      const candidateModelClasses = Array.from(new Set(
        (Array.isArray(report?.candidate_model_classes) ? report.candidate_model_classes : [])
          .map(text)
          .filter(Boolean),
      ));
      const source = reportSource(event, report);
      const key = [source.eventId || "", source.eventAt, topic, category, reportedClass, candidateModelClasses.join("\u0000")].join("\u0001");
      if (seen.has(key)) continue;
      seen.add(key);
      reports.push({ topic, category, reportedClass, candidateModelClasses, ...source });
    }
  }
  return reports;
}
