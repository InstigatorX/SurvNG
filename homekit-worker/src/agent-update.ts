import { execFile, spawn } from 'node:child_process';
import { once } from 'node:events';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { promisify } from 'node:util';
import { join } from 'node:path';

async function listing(file: string, verbose: boolean, signal?: AbortSignal): Promise<string[]> {
  const child = spawn('tar', [verbose ? '-tvzf' : '-tzf', file, '--numeric-owner'], {
    stdio: ['ignore', 'pipe', 'ignore'], timeout: 30000, killSignal: 'SIGKILL', signal });
  const chunks: Buffer[] = [];
  let size = 0;
  child.stdout.on('data', chunk => {
    size += chunk.length;
    if (size > 8 * 1024 * 1024) child.kill('SIGKILL'); else chunks.push(chunk);
  });
  const [code] = await once(child, 'close');
  if (code !== 0 || size > 8 * 1024 * 1024) throw new Error('update rejected');
  return Buffer.concat(chunks).toString('utf8').split('\n').filter(Boolean);
}

export async function validateArchive(file: string, allow: (name: string) => boolean, signal?: AbortSignal): Promise<string[]> {
  const names = await listing(file, false, signal);
  if (!names.length || new Set(names).size !== names.length || names.some(name => name.startsWith('/') || name.includes('\\') || name.split('/').includes('..') || !allow(name))) {
    throw new Error('update rejected');
  }
  // Names alone do not confine links or device entries to the private staging directory.
  const types = await listing(file, true, signal);
  if (types.length !== names.length || types.some(line => line[0] !== '-' && line[0] !== 'd')) throw new Error('update rejected');
  return names;
}

const ffmpegMembers = ['ffmpeg', 'ffprobe'];
const runtimeMembers = ['libva.so.2', 'libva-drm.so.2', 'libigdgmm.so.12', 'libmfx-gen.so.1.2', 'libvpl.so.2', 'dri/iHD_drv_video.so'];

export function toolMembers(kind: 'ffmpeg' | 'runtime'): readonly string[] {
  return kind === 'ffmpeg' ? ffmpegMembers : runtimeMembers;
}

export async function verifyToolRelease(kind: 'ffmpeg' | 'runtime', release: string, directory: string, signal: AbortSignal): Promise<void> {
  const digest = createHash('sha256');
  for (const name of toolMembers(kind)) {
    if (kind === 'runtime') digest.update(name);
    for await (const chunk of createReadStream(join(directory, name), { signal })) digest.update(chunk);
  }
  if (digest.digest('hex') !== release) throw new Error('HomeKit tool release mismatch');
}

export async function verifyEncoderTools(directory: string, signal?: AbortSignal): Promise<void> {
  const env = { ...process.env, LD_LIBRARY_PATH: join(directory, 'lib'), LIBVA_DRIVERS_PATH: join(directory, 'lib', 'dri') };
  const options = { cwd: directory, env, signal, timeout: 5000, maxBuffer: 512 * 1024, killSignal: 'SIGKILL' as const };
  const run = promisify(execFile);
  const ffmpeg = join(directory, 'bin', 'ffmpeg');
  await run(ffmpeg, ['-version'], options);
  await run(join(directory, 'bin', 'ffprobe'), ['-version'], options);
  const { stdout } = await run(ffmpeg, ['-hide_banner', '-encoders'], options);
  const encoders = new Set(stdout.split('\n').map(line => line.trim().split(/\s+/)[1]));
  if (['libx265', 'libopus', 'aac', 'mjpeg'].some(name => !encoders.has(name))) {
    throw new Error('staged FFmpeg lacks required encoders');
  }
}
