import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tiers, stopChild, type Tier } from './media.js';
import type { MediaAdmission } from './camera-media.js';
import { LowDelayProof } from './low-delay.js';

export interface VideoEncoding { backend: 'qsv' | 'software'; device?: string; fallbackReason?: 'qsv_unavailable' }
export function deviceArgs(encoding: VideoEncoding): string[] {
  return encoding.backend === 'qsv' ? ['-qsv_device', encoding.device!] : [];
}
const FAMILIES = [
  { ratio: 16 / 9, sizes: [[1920, 1080], [1280, 720], [640, 360]] },
  { ratio: 4 / 3, sizes: [[1600, 1200], [1440, 1080], [640, 480]] },
  { ratio: 9 / 16, sizes: [[1080, 1920], [720, 1280], [360, 640]] },
  { ratio: 3 / 4, sizes: [[1200, 1600], [960, 1280], [480, 640]] },
  { ratio: 1, sizes: [[1440, 1440], [1080, 1080], [480, 480]] },
] as const;
const TIER_RATES = [
  { identifier: 1, quality: 2, frameRate: 30, targetAverageBitrate: 1700, peakBitrate: 1800 },
  { identifier: 2, quality: 3, frameRate: 30, targetAverageBitrate: 768, peakBitrate: 800 },
  { identifier: 3, quality: 4, frameRate: 15, targetAverageBitrate: 180, peakBitrate: 190 },
] as const;

/** Required HomeKit resolutions for the camera's real aspect. Padding a 4:3 or 3:4 picture into 16:9 is what draws the side bars. */
export function streamProfile(width: number, height: number): { width: number; height: number; tiers: Tier[]; resolutions: [number, number, number][] } {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2) throw new Error('invalid frame');
  const ratio = Math.log(width / height);
  const family = FAMILIES.reduce((best, item) => Math.abs(Math.log(item.ratio) - ratio) < Math.abs(Math.log(best.ratio) - ratio) ? item : best);
  const selected = family.sizes.map((size, index) => ({ ...TIER_RATES[index], width: size[0], height: size[1] }));
  return { width, height, tiers: selected, resolutions: [
    [selected[0].width, selected[0].height, 30], [selected[0].width, selected[0].height, 15],
    [selected[1].width, selected[1].height, 30], [selected[1].width, selected[1].height, 15],
    [selected[2].width, selected[2].height, 15],
  ] };
}

export function encodeVideo(tier: Tier, encoding: VideoEncoding): string[] {
  // Fill the tier. decrease+pad keeps unused 16:9 margins and Home shows them as side bars.
  const filter = `scale=${tier.width}:${tier.height}:force_original_aspect_ratio=increase,crop=${tier.width}:${tier.height},setsar=1,fps=${tier.frameRate}`;
  const rate = ['-b:v', `${tier.targetAverageBitrate}k`, '-maxrate', `${tier.peakBitrate}k`, '-bufsize', `${tier.peakBitrate * 2}k`];
  if (encoding.backend === 'qsv') return ['-vf', `${filter},format=nv12`, '-c:v', 'hevc_qsv',
    '-preset', 'veryfast', '-profile:v', 'main', '-g', String(tier.frameRate * 4), '-bf', '0',
    '-forced_idr', '1', '-idr_interval', '0', '-gpb', '0', '-adaptive_i', '0', '-adaptive_b', '0',
    '-async_depth', '1', '-look_ahead_depth', '0', ...rate];
  return ['-vf', filter, '-c:v', 'libx265', '-preset', 'ultrafast', '-tune', 'zerolatency', '-pix_fmt', 'yuv420p', ...rate,
    '-x265-params', `pools=2:frame-threads=2:keyint=${tier.frameRate * 4}:min-keyint=${tier.frameRate * 4}:open-gop=0:scenecut=0:bframes=0:repeat-headers=1:log-level=error`];
}

/** Qualify real HEVC encoding AND decoding; an encoder listing alone is insufficient. */
export async function qualifyEncoding(ffmpeg: string, directory: string, admission: MediaAdmission,
  signal: AbortSignal, encoding: VideoEncoding): Promise<boolean> {
  signal.throwIfAborted();
  await admission('acquire', 'qsv-qualification', { purpose: 'encoder' });
  let root: string | undefined;
  const run = async (args: string[], proof?: LowDelayProof): Promise<boolean> => {
    signal.throwIfAborted();
    const child = spawn(ffmpeg, ['-hide_banner', '-loglevel', proof ? 'info' : 'error', '-nostdin', ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    child.stdout.on('data', (chunk: Buffer) => proof?.push('packets', chunk));
    child.stderr.on('data', (chunk: Buffer) => proof?.push('headers', chunk));
    const closed = once(child, 'close'); void closed.catch(() => {});
    const abort = () => { void stopChild(child); };
    signal.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(() => child.kill('SIGKILL'), 15000);
    try {
      if (child.pid) await admission('started', 'qsv-qualification', { pid: child.pid });
      const [code] = await closed;
      signal.throwIfAborted();
      return code === 0;
    } finally { clearTimeout(timer); signal.removeEventListener('abort', abort); await stopChild(child); }
  };
  try {
    root = await mkdtemp(join(directory, 'qsv-probe-'));
    const path = join(root, 'probe.mp4');
    const frames = tiers[0].frameRate * 4 + 5;
    if (!await run([...deviceArgs(encoding), '-f', 'lavfi', '-i', 'testsrc2=size=1920x1080:rate=30',
      '-frames:v', String(frames), ...encodeVideo(tiers[0], encoding), '-tag:v', 'hvc1', '-f', 'mp4', path])) return false;
    const proof = new LowDelayProof();
    if (!await run(['-i', path, '-map', '0:v:0', '-c:v', 'copy', '-bsf:v', 'hevc_mp4toannexb,trace_headers', '-f', 'framehash', 'pipe:1'], proof) || !proof.valid(frames)) return false;
    return await run(['-i', path, '-f', 'null', '-']);
  } catch (error) {
    signal.throwIfAborted();
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
    throw error;
  } finally {
    try { if (root) await rm(root, { recursive: true, force: true }); }
    finally { await admission('release', 'qsv-qualification'); }
  }
}

export class EncodingUnavailableError extends Error {}

export async function selectEncoding(allowSoftware: boolean, qualify: () => Promise<boolean>, qualifySoftware: () => Promise<boolean>): Promise<VideoEncoding> {
  if (await qualify()) return { backend: 'qsv', device: '/dev/dri/renderD128' };
  if (!allowSoftware || !await qualifySoftware()) throw new EncodingUnavailableError('No permitted HEVC encoder is usable');
  return { backend: 'software', fallbackReason: 'qsv_unavailable' };
}
