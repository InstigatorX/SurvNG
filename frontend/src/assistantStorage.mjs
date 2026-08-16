export const ASSISTANT_HISTORY_TTL_MS = 24 * 60 * 60 * 1000;

export function readAssistantHistory(storage, key, now = Date.now()) {
  try {
    const parsed = JSON.parse(storage?.getItem(key) || "null");
    if (!parsed || ![1, 2].includes(parsed.version) || !Array.isArray(parsed.messages)) return [];
    if (!Number.isFinite(parsed.expires_at) || parsed.expires_at <= now) {
      storage?.removeItem(key);
      return [];
    }
    return parsed.messages.slice(-30).map((message) => ({
      ...message,
      ...(message.context && typeof message.context === "object" ? { context: message.context } : {}),
    }));
  } catch {
    return [];
  }
}

export function writeAssistantHistory(storage, key, messages, now = Date.now()) {
  try {
    storage?.setItem(key, JSON.stringify({
      version: 2,
      expires_at: now + ASSISTANT_HISTORY_TTL_MS,
      messages: Array.isArray(messages) ? messages.slice(-30) : [],
    }));
    return true;
  } catch {
    return false;
  }
}
