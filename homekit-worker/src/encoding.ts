import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { tiers, stopChild, type Tier } from './media.js';
import type { MediaAdmission } from './camera-media.js';
import { LowDelayProof } from './low-delay.js';

export interface VideoEncoding { backend: 'qsv' | 'software'; device?: string; fallbackReason?: 'qsv_unavailable' }
export function deviceArgs(encoding: VideoEncoding): string[] {
  return encoding.backend === 'qsv' ? ['-qsv_device', encoding.device!] : [];
}
// Every tier fits the cellular relay envelope (Scrypted #2152: about 720p and an
// 800 kbit/s peak accepted, Full HD and a 1.078 Mbit/s relay cap rejected). A home
// hub relays remote viewers over the local RTP path using the tier it requests,
// and in real-camera testing it requested the top tier, so the top tier must fit.
const FAMILIES = [
  { ratio: 16 / 9, sizes: [[1280, 720], [960, 540], [640, 360]] },
  { ratio: 4 / 3, sizes: [[960, 720], [800, 600], [640, 480]] },
  { ratio: 9 / 16, sizes: [[720, 1280], [540, 960], [360, 640]] },
  { ratio: 3 / 4, sizes: [[720, 960], [600, 800], [480, 640]] },
  { ratio: 1, sizes: [[720, 720], [600, 600], [480, 480]] },
] as const;
const TIER_RATES = [
  { identifier: 1, quality: 2, frameRate: 30, targetAverageBitrate: 768, peakBitrate: 800 },
  { identifier: 2, quality: 3, frameRate: 30, targetAverageBitrate: 480, peakBitrate: 500 },
  { identifier: 3, quality: 4, frameRate: 15, targetAverageBitrate: 180, peakBitrate: 190 },
] as const;

export function selectCellularTier(catalog: readonly Tier[]): Tier | undefined {
  // Match the bitrate and dimensions Apple accepted on the cellular relay in Scrypted #2152.
  return catalog.filter(({ peakBitrate, width, height }) =>
    peakBitrate <= 800 && Math.max(width, height) <= 1280 && Math.min(width, height) <= 720)
    .sort((a, b) => b.peakBitrate - a.peakBitrate)[0];
}

export function recordingProfile(profile: { tiers: Tier[]; resolutions: [number, number, number][] }, shareCellularTier: boolean) {
  const sharedTier = shareCellularTier ? selectCellularTier(profile.tiers) : undefined;
  if (!sharedTier) return { ...profile, sharedTier };

  const { width, height, frameRate } = sharedTier;
  // Home requires a 15 fps choice. The recording delegate can use the same-size
  // producer when Home selects 15 fps for a 30 fps tier.
  const resolutions: [number, number, number][] = [[width, height, frameRate]];
  if (frameRate !== 15) resolutions.push([width, height, 15]);
  return { tiers: [sharedTier], resolutions, sharedTier };
}

/** Fit the camera's aspect inside each tier's encoding budget, including wide cameras. */
export function streamProfile(width: number, height: number): { width: number; height: number; tiers: Tier[]; resolutions: [number, number, number][] } {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2) throw new Error('invalid frame');
  const ratio = Math.log(width / height);
  const family = FAMILIES.reduce((best, item) => Math.abs(Math.log(item.ratio) - ratio) < Math.abs(Math.log(best.ratio) - ratio) ? item : best);
  const selected = family.sizes.map((size, index) => {
    const scale = Math.min(size[0] / width, size[1] / height);
    return { ...TIER_RATES[index],
      width: Math.max(2, Math.min(size[0], Math.round(width * scale / 2) * 2)),
      height: Math.max(2, Math.min(size[1], Math.round(height * scale / 2) * 2)) };
  });
  return { width, height, tiers: selected, resolutions: [
    [selected[0].width, selected[0].height, 30], [selected[0].width, selected[0].height, 15],
    [selected[1].width, selected[1].height, 30], [selected[1].width, selected[1].height, 15],
    [selected[2].width, selected[2].height, 15],
  ] };
}

export async function probeFrame(ffmpeg: string, url: string, transport: 'tcp' | 'udp', signal?: AbortSignal): Promise<{ width: number; height: number }> {
  signal?.throwIfAborted();
  const args = ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,sample_aspect_ratio', '-of', 'json'];
  if (/^rtsps?:/i.test(url)) args.unshift('-rtsp_transport', transport);
  const child = spawn(join(dirname(ffmpeg), 'ffprobe'), [...args, url], { stdio: ['ignore', 'pipe', 'ignore'], signal });
  const chunks: Buffer[] = [];
  let size = 0;
  child.stdout!.on('data', (chunk: Buffer) => {
    size += chunk.length;
    if (size > 65536) child.kill('SIGKILL'); else chunks.push(chunk);
  });
  const timer = setTimeout(() => child.kill('SIGKILL'), 15000);
  try {
    const [code] = await once(child, 'close');
    signal?.throwIfAborted();
    if (code !== 0 || size > 65536) throw new Error('camera frame probe failed');
    const stream = JSON.parse(Buffer.concat(chunks).toString('utf8'))?.streams?.[0];
    const width = Number(stream?.width), height = Number(stream?.height);
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2) throw new Error('camera frame probe failed');
    const sar = /^(\d+):(\d+)$/.exec(String(stream?.sample_aspect_ratio ?? ''));
    if (!sar || !Number(sar[1]) || !Number(sar[2])) return { width, height };
    return { width: Math.max(2, Math.round(width * Number(sar[1]) / Number(sar[2]) / 2) * 2), height };
  } finally { clearTimeout(timer); await stopChild(child); }
}

export function encodeVideo(tier: Tier, encoding: VideoEncoding): string[] {
  // Preserve the whole field of view. DAR handles non-square input pixels;
  // padding covers only the mismatch from even-pixel tier rounding or a source
  // resolution change after probing. Never enlarge-and-crop camera footage.
  const filter = `scale=w='max(2,trunc(min(${tier.width},${tier.height}*dar)/2)*2)':h='max(2,trunc(min(${tier.height},${tier.width}/dar)/2)*2)',setsar=1,pad=${tier.width}:${tier.height}:(ow-iw)/2:(oh-ih)/2,fps=${tier.frameRate}`;
  const rate = ['-b:v', `${tier.targetAverageBitrate}k`, '-maxrate', `${tier.peakBitrate}k`, '-bufsize', `${tier.peakBitrate * 2}k`];
  if (encoding.backend === 'qsv') return ['-vf', `${filter},format=nv12`, '-c:v', 'hevc_qsv',
    '-preset', 'veryfast', '-profile:v', 'main', '-g', String(tier.frameRate * 4), '-bf', '0',
    '-forced_idr', '1', '-idr_interval', '0', '-gpb', '0', '-adaptive_i', '0', '-adaptive_b', '0',
    '-async_depth', '1', '-look_ahead_depth', '0', '-fps_mode', 'cfr',
    '-bsf:v', 'dump_extra=freq=keyframe', ...rate];
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
