import { isIPv4 } from 'node:net';

/** Streaming camera uploads cannot be replayed or redirected. Undici's default
 * redirect-follow path clones/tees the body and retains the unread branch for
 * the lifetime of an upload, which is unbounded for a live camera feed.
 */
export function post(url: string, body: BodyInit | ReadableStream | NodeJS.ReadableStream,
  headers: Record<string, string>, signal: AbortSignal, streaming = false): Promise<Response> {
  return fetch(url, { method: 'POST', headers, body: body as BodyInit, signal,
    ...(streaming ? { duplex: 'half', redirect: 'error' } : {}) } as RequestInit);
}

/** The primary advertises either an HTTPS override or its private IPv4 listener. */
export function mediaUploadOrigin(enrolled: string, override: unknown): string {
  if (override == null) return enrolled;
  const parts = typeof override === 'string' ? /^(https?):\/\/([^/\s\\?#@]+)((?:\/[A-Za-z0-9._~-]+)*)\/?$/i.exec(override) : null;
  if (!parts || parts[0] !== override || parts[3].split('/').some(segment => segment === '.' || segment === '..')) {
    throw new Error('invalid HomeKit media endpoint');
  }
  const selected = new URL(parts[0]);
  if (selected.protocol === 'http:') {
    const host = parts[2].split(':')[0];
    const [first, second] = host.split('.').map(Number);
    if (!isIPv4(host) || selected.port === '0' ||
        !(first === 10 || (first === 172 && second >= 16 && second <= 31) || (first === 192 && second === 168))) {
      throw new Error('invalid HomeKit LAN media endpoint');
    }
  }
  return selected.href.replace(/\/$/, '');
}
