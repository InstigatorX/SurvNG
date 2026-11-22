/** Streaming camera uploads cannot be replayed or redirected. Undici's default
 * redirect-follow path clones/tees the body and retains the unread branch for
 * the lifetime of an upload, which is unbounded for a live camera feed.
 */
export function post(url: string, body: BodyInit | ReadableStream | NodeJS.ReadableStream,
  headers: Record<string, string>, signal: AbortSignal, streaming = false): Promise<Response> {
  return fetch(url, { method: 'POST', headers, body: body as BodyInit, signal,
    ...(streaming ? { duplex: 'half', redirect: 'error' } : {}) } as RequestInit);
}

/** Resolve each sync independently so removing an override restores the enrolled URL. */
export function mediaUploadOrigin(enrolled: string, override: unknown): string {
  if (override == null) return enrolled;
  const parts = typeof override === 'string' ? /^https:\/\/([^/\s\\?#@]+)((?:\/[A-Za-z0-9._~-]+)*)\/?$/i.exec(override) : null;
  if (!parts || parts[0] !== override || parts[2].split('/').some(segment => segment === '.' || segment === '..')) {
    throw new Error('invalid HomeKit media endpoint');
  }
  const selected = new URL(parts[0]);
  return selected.href.replace(/\/$/, '');
}
