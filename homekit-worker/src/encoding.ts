import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tiers, stopChild, type Tier } from './media.js';
import type { MediaAdmission } from './camera-media.js';

export interface VideoEncoding { backend: 'qsv' | 'software'; device?: string; fallbackReason?: 'qsv_unavailable' }
export function deviceArgs(encoding: VideoEncoding): string[] {
  return encoding.backend === 'qsv' ? ['-qsv_device', encoding.device!] : [];
}
export function encodeVideo(tier: Tier, encoding: VideoEncoding): string[] {
  const filter = `scale=${tier.width}:${tier.height}:force_original_aspect_ratio=decrease,pad=${tier.width}:${tier.height}:(ow-iw)/2:(oh-ih)/2,fps=${tier.frameRate}`;
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
  const run = async (args: string[]): Promise<boolean> => {
    signal.throwIfAborted();
    const child = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-nostdin', ...args], { stdio: ['ignore', 'ignore', 'ignore'] });
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
    const path = join(root, 'probe.hevc');
    if (!await run([...deviceArgs(encoding), '-f', 'lavfi', '-i', 'testsrc2=size=1920x1080:rate=30',
      '-frames:v', '2', ...encodeVideo(tiers[0], encoding), '-f', 'hevc', path])) return false;
    return await run(['-i', path, '-frames:v', '2', '-f', 'null', '-']);
  } catch (error) {
    signal.throwIfAborted();
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return false;
    throw error;
  } finally {
    if (root) await rm(root, { recursive: true, force: true });
    await admission('release', 'qsv-qualification');
  }
}

export class EncodingUnavailableError extends Error {}

export async function selectEncoding(allowSoftware: boolean, qualify: () => Promise<boolean>, qualifySoftware: () => Promise<boolean>): Promise<VideoEncoding> {
  if (await qualify()) return { backend: 'qsv', device: '/dev/dri/renderD128' };
  if (!allowSoftware || !await qualifySoftware()) throw new EncodingUnavailableError('No permitted HEVC encoder is usable');
  return { backend: 'software', fallbackReason: 'qsv_unavailable' };
}
