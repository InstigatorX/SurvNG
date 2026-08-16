export function safeMediaUrl(value, basePath = "", origin = "") {
  const url = String(value || "").trim();
  if (!url || url.startsWith("//")) return "";
  if (/^(blob:|data:image\/)/i.test(url)) return url;
  if (/^https?:\/\//i.test(url)) {
    try {
      const parsed = new URL(url);
      return origin && parsed.origin === origin ? url : "";
    } catch {
      return "";
    }
  }
  if (!url.startsWith("/")) return "";
  const base = String(basePath || "").replace(/\/+$/, "");
  if (!base || url === base || url.startsWith(`${base}/`)) return url;
  return `${base}${url}`;
}
