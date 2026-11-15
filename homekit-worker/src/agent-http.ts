/** Streaming camera uploads cannot be replayed or redirected. Undici's default
 * redirect-follow path clones/tees the body and retains the unread branch for
 * the lifetime of an upload, which is unbounded for a live camera feed.
 */
export function post(url: string, body: BodyInit | ReadableStream | NodeJS.ReadableStream,
  headers: Record<string, string>, signal: AbortSignal, streaming = false): Promise<Response> {
  return fetch(url, { method: 'POST', headers, body: body as BodyInit, signal,
    ...(streaming ? { duplex: 'half', redirect: 'error' } : {}) } as RequestInit);
}
