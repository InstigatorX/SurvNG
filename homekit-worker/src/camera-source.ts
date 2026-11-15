import { createServer, type Socket } from 'node:net';
import { chmod } from 'node:fs/promises';
import { once } from 'node:events';
import type { ChildProcess } from 'node:child_process';
import { stopChild } from './media.js';

const GOP_LIMIT = 2 * 1024 * 1024;

function packetPayload(packet: Buffer): Buffer {
  const adapt = (packet[3] >> 4) & 0x3;
  if (adapt === 2) return Buffer.alloc(0);
  let offset = 4;
  if (adapt === 3) offset += 1 + packet[4];
  return offset < packet.length ? packet.subarray(offset) : Buffer.alloc(0);
}

function psi(body: Buffer, pusi: boolean): Buffer {
  if (!pusi || body.length === 0) return Buffer.alloc(0);
  const start = 1 + body[0];
  return start < body.length ? body.subarray(start) : Buffer.alloc(0);
}

function patProgram(table: Buffer): number {
  if (table.length < 12 || table[0] !== 0) return -1;
  const length = ((table[1] & 0x0f) << 8) | table[2];
  const end = Math.min(table.length, 3 + length - 4);
  for (let offset = 8; offset + 4 <= end; offset += 4) {
    const program = (table[offset] << 8) | table[offset + 1];
    const pid = ((table[offset + 2] & 0x1f) << 8) | table[offset + 3];
    if (program !== 0) return pid;
  }
  return -1;
}

function pmtVideo(table: Buffer): number {
  if (table.length < 12 || table[0] !== 2) return -1;
  const info = ((table[10] & 0x0f) << 8) | table[11];
  let offset = 12 + info;
  const length = ((table[1] & 0x0f) << 8) | table[2];
  const end = Math.min(table.length, 3 + length - 4);
  while (offset + 5 <= end) {
    const streamType = table[offset];
    const pid = ((table[offset + 1] & 0x1f) << 8) | table[offset + 2];
    const infoLength = ((table[offset + 3] & 0x0f) << 8) | table[offset + 4];
    if (streamType === 0x24) return pid;
    offset += 5 + infoLength;
  }
  return -1;
}

/** True when this PES begins a HEVC keyframe (parameter sets or an IDR). */
function elementaryKey(body: Buffer): boolean {
  if (body.length < 9 || body[0] !== 0 || body[1] !== 0 || body[2] !== 1) return false;
  if (body[3] < 0xe0 || body[3] > 0xef) return false;
  const start = 9 + body[8];
  const window = body.subarray(start, Math.min(body.length, start + 80));
  for (let i = 0; i + 3 < window.length; i++) {
    let nalAt = -1;
    if (window[i] === 0 && window[i + 1] === 0 && window[i + 2] === 1) nalAt = i + 3;
    else if (i + 4 < window.length && window[i] === 0 && window[i + 1] === 0 && window[i + 2] === 0 && window[i + 3] === 1) nalAt = i + 4;
    if (nalAt < 0 || nalAt >= window.length) continue;
    const nal = (window[nalAt] >> 1) & 0x3f;
    if (nal === 19 || nal === 20 || nal === 21 || nal === 32) return true;
    if (nal === 35 || nal === 39) continue;
    return false;
  }
  return false;
}

/** Remembers the current HEVC GOP so a new viewer can start at its keyframe. */
export class TsReplay {
  private rest: Buffer<ArrayBufferLike> = Buffer.alloc(0);
  private gop: Buffer<ArrayBufferLike>[] = [];
  private gopBytes = 0;
  private pat: Buffer<ArrayBufferLike> = Buffer.alloc(0);
  private pmt: Buffer<ArrayBufferLike> = Buffer.alloc(0);
  private pmtPid = -1;
  private videoPid = -1;

  push(chunk: Buffer): void {
    if (chunk.length === 0) return;
    this.rest = this.rest.length === 0 ? Buffer.from(chunk) : Buffer.concat([this.rest, chunk]);
    let offset = 0;
    while (this.rest.length - offset >= 188) {
      if (this.rest[offset] !== 0x47) {
        const next = this.rest.indexOf(0x47, offset + 1);
        if (next < 0) { this.rest = Buffer.alloc(0); return; }
        offset = next;
        continue;
      }
      this.take(Buffer.from(this.rest.subarray(offset, offset + 188)));
      offset += 188;
    }
    this.rest = offset === 0 ? this.rest : Buffer.from(this.rest.subarray(offset));
  }

  private take(packet: Buffer): void {
    const pid = ((packet[1] & 0x1f) << 8) | packet[2];
    const pusi = (packet[1] & 0x40) !== 0;
    const body = packetPayload(packet);
    if (pid === 0 && pusi) {
      const program = patProgram(psi(body, true));
      if (program > 0) this.pmtPid = program;
      this.pat = packet;
    } else if (this.pmtPid === pid && pusi) {
      const video = pmtVideo(psi(body, true));
      if (video > 0) this.videoPid = video;
      this.pmt = packet;
    }
    if (pusi && (this.videoPid < 0 || pid === this.videoPid) && elementaryKey(body)) {
      this.videoPid = pid;
      this.gop = [packet];
      this.gopBytes = packet.length;
      return;
    }
    if (this.gopBytes === 0) return;
    this.gop.push(packet);
    this.gopBytes += packet.length;
    if (this.gopBytes > GOP_LIMIT) { this.gop = []; this.gopBytes = 0; }
  }

  /** PAT, PMT, and every packet since the latest keyframe. Empty before the first one. */
  snapshot(): Buffer {
    if (this.gopBytes === 0) return Buffer.alloc(0);
    // Live writes retain the producer's chunk boundaries. Include any partial
    // packet already broadcast so the next live chunk completes it for a joiner.
    return Buffer.concat([this.pat, this.pmt, ...this.gop, this.rest]);
  }
}

/** One normalized HEVC/AAC producer per requested tier, shared by consumers.
 * A new consumer is handed the keyframe already in flight, then live packets.
 * Nothing is written to disk, and a slow consumer is disconnected on its own.
 */
export class CameraSource {
  private replay = new TsReplay();
  private server = createServer(client => {
    if (this.closed) { client.destroy(); return; }
    client.on('close', () => this.clients.delete(client));
    client.on('error', () => client.destroy());
    const prior = this.replay.snapshot();
    if (prior.length > 0) client.write(prior);
    this.clients.add(client);
    void (this.producing ??= this.produce()).catch(() => { void this.stop(); });
  });
  private clients = new Set<Socket>();
  private child?: ChildProcess;
  private starting?: Promise<string>;
  private producing?: Promise<void>;
  private stopping?: Promise<void>;
  private watchdog?: NodeJS.Timeout;
  closed = false;
  references = 0;

  constructor(private path: string, private producer: () => Promise<ChildProcess>) {}
  start(): Promise<string> { return this.starting ??= this.open(); }
  private async open(): Promise<string> {
    this.server.listen(this.path); await once(this.server, 'listening');
    await chmod(this.path, 0o600);
    return `unix://${this.path}`;
  }
  private async produce(): Promise<void> {
    try {
      this.child = await this.producer();
      if (this.closed) { await stopChild(this.child); return; }
      let lastData = Date.now();
      this.child.stdout!.on('data', (data: Buffer) => {
        lastData = Date.now();
        this.replay.push(data);
        for (const client of this.clients) {
          // Bound each slow consumer independently; never block the producer.
          if (client.writableLength > 2 * 1024 * 1024) client.destroy();
          else client.write(data);
        }
      });
      this.child.on('error', () => { void this.stop(); });
      this.child.on('close', () => { void this.stop(); });
      this.watchdog = setInterval(() => { if (Date.now() - lastData > 15000) void this.stop(); }, 1000);
    } catch (error) { this.closed = true; throw error; }
  }
  stop(): Promise<void> {
    this.closed = true;
    return this.stopping ??= (async () => {
      clearInterval(this.watchdog);
      for (const client of this.clients) client.destroy();
      await this.producing?.catch(() => {});
      if (this.child) await stopChild(this.child);
      await new Promise<void>((resolve, reject) => this.server.close(error => {
        if (error && (error as NodeJS.ErrnoException).code !== 'ERR_SERVER_NOT_RUNNING') reject(error); else resolve();
      }));
    })();
  }
}
