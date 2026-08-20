import { SECRET_PLACEHOLDER } from "./constants.js";

export function clearMaskedUrlPassword(value) {
  if (!value || !value.includes(SECRET_PLACEHOLDER)) return value || "";
  return value.replace(`:${SECRET_PLACEHOLDER}@`, "@");
}

export function clearMaskedSecret(value) {
  return value === SECRET_PLACEHOLDER ? "" : value || "";
}

export function secretInputValue(value) {
  return value === SECRET_PLACEHOLDER ? "" : value || "";
}

export function secretInputHint(value, fallback = "") {
  return value === SECRET_PLACEHOLDER ? "Saved — type to replace" : fallback;
}
