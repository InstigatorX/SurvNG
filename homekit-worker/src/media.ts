import { spawn, type ChildProcess } from 'node:child_process';
import { createSocket, type Socket } from 'node:dgram';
import { once } from 'node:events';
import { mkdir, stat, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { RtpPacket } from 'werift';

export const tiers = [
  { identifier: 1, quality: 2, width: 1920, height: 1080, frameRate: 30, targetAverageBitrate: 1700, peakBitrate: 1800 },
  { identifier: 2, quality: 3, width: 1280, height: 720, frameRate: 30, targetAverageBitrate: 768, peakBitrate: 800 },
  { identifier: 3, quality: 4, width: 640, height: 360, frameRate: 15, targetAverageBitrate: 180, peakBitrate: 190 },
];
export type Tier = typeof tiers[number];

/** The lab only accepts generated fixtures; it cannot open a production camera. */
export class SyntheticMedia {
  readonly children = new Set<ChildProcess>();
  constructor(readonly directory: string, readonly ffmpeg = 'ffmpeg') {}
  path(tier: Tier): string { return join(this.directory, `synthetic-${tier.identifier}.mp4`); }

  async prepare(): Promise<void> {
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    for (const tier of tiers) {
      const file = this.path(tier);
      try { if ((await stat(file)).size > 0) continue; } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      }
      // One-time fixture encoding, never a camera URL. Deterministic 2-second closed GOPs.
      const temporary = `${file}.partial.mp4`;
      const child = this.start(['-f', 'lavfi', '-i', `testsrc2=size=${tier.width}x${tier.height}:rate=${tier.frameRate}`,
        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=32000', '-t', '8',
        '-c:v', 'libx265', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p',
        '-b:v', `${tier.targetAverageBitrate}k`, '-maxrate', `${tier.peakBitrate}k`, '-bufsize', `${tier.peakBitrate * 2}k`,
        '-x265-params', `pools=2:frame-threads=2:keyint=${tier.frameRate * 2}:min-keyint=${tier.frameRate * 2}:scenecut=0:bframes=0:repeat-headers=1:log-level=error`,
        '-tag:v', 'hvc1', '-c:a', 'aac', '-ac', '1', '-ar', '32000', '-b:a', '32k', '-y', temporary]);
      const [code] = await once(child, 'close');
      if (code !== 0) throw new Error('synthetic fixture encoding failed');
      await rename(temporary, file);
    }
  }

  start(args: string[]): ChildProcess {
    const child = spawn(this.ffmpeg, ['-hide_banner', '-loglevel', 'warning', '-nostdin', ...args], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    this.children.add(child);
    child.on('close', () => this.children.delete(child));
    // Arguments contain synthetic files only; still keep stderr out of status/API payloads.
    let warnings = 0;
    child.stderr!.on('data', (chunk: Buffer) => {
      if (++warnings <= 4) console.error(`Synthetic FFmpeg: ${chunk.toString('utf8').slice(0, 1000).trim()}`);
    });
    child.on('close', () => { if (warnings > 4) console.error(`Synthetic FFmpeg: ${warnings - 4} additional diagnostic chunks suppressed.`); });
    child.on('error', () => console.error('FFmpeg process failed to start.'));
    return child;
  }

  input(tier: Tier): string[] { return ['-re', '-stream_loop', '-1', '-i', this.path(tier)]; }

  async snapshot(): Promise<Buffer> {
    const child = this.start(['-i', this.path(tiers[0]), '-frames:v', '1', '-an', '-c:v', 'mjpeg', '-f', 'image2pipe', 'pipe:1']);
    const chunks: Buffer[] = [];
    let size = 0;
    const timeout = setTimeout(() => child.kill('SIGKILL'), 10_000);
    child.stdout!.on('data', (chunk: Buffer) => { size += chunk.length; if (size > 4 * 1024 * 1024) child.kill('SIGKILL'); else chunks.push(chunk); });
    try {
      const [code] = await once(child, 'close');
      if (code !== 0 || size === 0 || size > 4 * 1024 * 1024) throw new Error('snapshot unavailable');
      return Buffer.concat(chunks);
    } finally { clearTimeout(timeout); }
  }

  async stop(): Promise<void> { await Promise.all([...this.children].map(stopChild)); }
}

export async function stopChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const closed = once(child, 'close');
  child.kill('SIGTERM');
  const timer = setTimeout(() => child.kill('SIGKILL'), 2000);
  try { await closed; } finally { clearTimeout(timer); }
}

export async function udp(address = '127.0.0.1'): Promise<Socket> {
  const socket = createSocket(address.includes(':') ? 'udp6' : 'udp4');
  socket.bind(0, address);
  try { await once(socket, 'listening'); return socket; }
  catch (error) { socket.close(); throw error; }
}

export class RtpFeed {
  private child?: ChildProcess;
  private sockets: Socket[] = [];
  private stopped = false;
  constructor(private media: SyntheticMedia, private tier: Tier) {}
  async start(video: (packet: RtpPacket) => void, audio: (packet: RtpPacket) => void, failed: () => void): Promise<void> {
    try {
      for (const handler of [video, audio]) {
        const socket = await udp();
        this.sockets.push(socket);
        socket.on('message', data => { try { handler(RtpPacket.deSerialize(data)); } catch { failed(); } });
        socket.on('error', failed);
      }
      if (this.stopped) { await this.stop(); return; }
      const [v, a] = this.sockets.map(socket => socket.address().port);
      this.child = this.media.start([...this.media.input(this.tier),
        '-map', '0:v:0', '-c:v', 'copy', '-an', '-payload_type', '99', '-f', 'rtp', `rtp://127.0.0.1:${v}?pkt_size=1200`,
        '-map', '0:a:0', '-c:a', 'libopus', '-ac', '1', '-ar', '48000', '-b:a', '24k', '-frame_duration', '20',
        '-vn', '-payload_type', '110', '-f', 'rtp', `rtp://127.0.0.1:${a}?pkt_size=1200`]);
      this.child.stdout!.resume();
      this.child.on('error', failed);
      this.child.on('close', () => { if (!this.stopped) failed(); });
    } catch (error) { await this.stop(); throw error; }
  }
  async stop(): Promise<void> {
    this.stopped = true;
    if (this.child) await stopChild(this.child);
    for (const socket of this.sockets.splice(0)) socket.close();
  }
}
