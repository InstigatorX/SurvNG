import React from "react";
import {
  UserRound,
} from "lucide-react";

export function visibleIdentity(item) {
  const identities = Array.isArray(item?.identities) ? item.identities : [];
  const faces = Array.isArray(item?.faces) ? item.faces : [];
  const candidates = identities.length ? identities : faces;

  const confirmed = candidates.find((identity) => (
    identity?.status === "confirmed"
    && String(identity?.name || "").trim()
  ));
  if (confirmed) return {
    name: String(confirmed.name).trim(),
    unknown: false,
    automatic: false,
    status: "confirmed",
    confidence: Number(confirmed.confidence) || 0,
  };

  const automatic = candidates.find((identity) => (
    identity?.status === "automatic"
    && String(identity?.name || "").trim()
  ));
  if (automatic) return {
    name: String(automatic.name).trim(),
    unknown: false,
    automatic: true,
    status: "automatic",
    confidence: Number(automatic.confidence) || 0,
  };

  const unknown = candidates.find((identity) => (
    identity?.status === "unknown"
    && (
      Number(identity?.unknown_cluster_id) > 0
      || /^Unknown Person \d+$/i.test(String(identity?.name || ""))
    )
  ));
  if (!unknown) return null;

  const clusterId = Number(unknown.unknown_cluster_id) || 0;
  return {
    name: clusterId > 0
      ? `Unknown Person ${clusterId}`
      : String(unknown.name || "Unknown person"),
    unknown: true,
    automatic: false,
    status: "unknown",
    confidence: Number(unknown.confidence) || 0,
  };
}

export function IdentityChip({ item, className = "" }) {
  const identity = visibleIdentity(item);
  if (!identity) return null;
  return (
    <span
      className={`identity-chip ${identity.unknown ? "unknown" : identity.automatic ? "automatic" : "known"} ${className}`.trim()}
      title={identity.confidence > 0
        ? `${identity.name} · ${Math.round(identity.confidence * 100)}% ${identity.automatic ? "automatic identity confidence" : "identity confidence"}`
        : identity.name}
    >
      <UserRound size={12} />
      {identity.name}
      {identity.automatic ? <span aria-label="Automatically identified">Auto</span> : null}
    </span>
  );
}
