import { spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import { createWriteStream, existsSync } from 'node:fs';
import { chmod, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { hostname, tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { encoderCommand } from './camera-media.js';
import { probeFrame, qualifyEncoding, selectEncoding, streamProfile, EncodingUnavailableError, type VideoEncoding } from './encoding.js';
import type { Tier } from './media.js';

const origin = process.env.SURVNG_ORIGIN?.replace(/\/$/, '');
let mediaOrigin = origin;
const secret = process.env.SURVNG_NODE_SECRET || '';
const home = process.cwd();
const bundledFfmpeg = join(home, 'bin', 'ffmpeg');
const ffmpeg = existsSync(bundledFfmpeg) ? bundledFfmpeg : (process.env.SURVNG_FFMPEG || 'ffmpeg');
const runtimeLib = join(home, 'lib');
if (existsSync(runtimeLib)) {
  process.env.LD_LIBRARY_PATH = [runtimeLib, process.env.LD_LIBRARY_PATH].filter(Boolean).join(':');
  process.env.LIBVA_DRIVERS_PATH = join(runtimeLib, 'dri');
}
if (!origin || !/^https?:\/\//i.test(origin) || !secret) {
  console.error('SurvNG HomeKit encoder configuration is incomplete.');
  process.exit(1);
}

type Assignment = { id: string; url: string; audio: boolean; transport: 'tcp' | 'udp'; allowSoftwareEncoding: boolean };
type Profile = ReturnType<typeof streamProfile>;
const profiles = new Map<string, Profile>();
const encodings = new Map<string, VideoEncoding | 'unavailable'>();
const rates = new Map<string, number>();
const feeds = new Map<string, ChildProcess>();

function devicePresent(): 'qsv' | 'software' {
  return existsSync('/dev/dri/renderD128') ? 'qsv' : 'software';
}

async function request(path: string, body: BodyInit | ReadableStream | NodeJS.ReadableStream, headers: Record<string, string>, stream = false, base = origin): Promise<Response> {
  return fetch(`${base}${path}`, { method: 'POST', headers: { authorization: `Bearer ${secret}`, ...headers }, body: body as BodyInit,
    ...(stream ? { duplex: 'half' } : {}) } as RequestInit);
}

function stopFeed(key: string): void {
  const child = feeds.get(key);
  if (!child) return;
  feeds.delete(key);
  child.kill('SIGTERM');
}

function startFeed(camera: Assignment, tier: Tier, encoding: VideoEncoding): void {
  const key = `${camera.id}:${tier.identifier}`;
  if (feeds.has(key)) return;
  const child = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-nostdin', '-progress', 'pipe:3', '-stats_period', '1',
    ...encoderCommand(camera.url, camera.transport, camera.audio, tier, encoding)], { stdio: ['ignore', 'pipe', 'ignore', 'pipe'] });
  feeds.set(key, child);
  child.once('close', () => { if (feeds.get(key) === child) feeds.delete(key); });
  let pending = '';
  child.stdio[3]?.on('data', (chunk: Buffer) => {
    pending += chunk.toString('utf8');
    if (pending.length > 8192) { pending = ''; return; }
    let end = pending.indexOf('\n');
    while (end >= 0) {
      const line = pending.slice(0, end);
      pending = pending.slice(end + 1);
      if (line.startsWith('fps=')) {
        const value = Number(line.slice(4));
        if (Number.isFinite(value)) rates.set(camera.id, value);
      }
      end = pending.indexOf('\n');
    }
  });
  void request('/api/homekit/nodes/media', child.stdout!, {
    'content-type': 'application/octet-stream', 'x-survng-camera': camera.id, 'x-survng-tier': String(tier.identifier),
  }, true, mediaOrigin).catch(() => stopFeed(key));
}

async function takeSnapshot(camera: Assignment): Promise<void> {
  const child = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-nostdin',
    ...( /^rtsps?:/i.test(camera.url) ? ['-rtsp_transport', camera.transport, '-timeout', '15000000'] : ['-rw_timeout', '15000000']),
    '-i', camera.url, '-map', '0:v:0', '-frames:v', '1', '-an', '-vf', 'scale=1280:-2', '-c:v', 'mjpeg', '-f', 'image2pipe', 'pipe:1'],
    { stdio: ['ignore', 'pipe', 'ignore'] });
  const chunks: Buffer[] = [];
  child.stdout!.on('data', (chunk: Buffer) => { if (chunks.reduce((sum, item) => sum + item.length, 0) < 4 * 1024 * 1024) chunks.push(chunk); });
  const timer = setTimeout(() => child.kill('SIGKILL'), 10000);
  const [code] = await new Promise<[number | null]>((resolve) => child.once('close', code => resolve([code])));
  clearTimeout(timer);
  if (code !== 0) return;
  await request('/api/homekit/nodes/snapshot', Buffer.concat(chunks), { 'content-type': 'image/jpeg', 'x-survng-camera': camera.id }).catch(() => {});
}

type Releases = { agent?: string; ffmpeg?: string; runtime?: string };
const releasesPath = join(home, '.releases.json');
let updateRetryAt = 0;

async function savedReleases(): Promise<Releases> {
  try { return JSON.parse(await readFile(releasesPath, 'utf8')) as Releases; }
  catch { return {}; }
}

async function pullArchive(path: string): Promise<string> {
  const response = await fetch(`${origin}${path}`, { headers: { authorization: `Bearer ${secret}` } });
  if (!response.ok || !response.body) throw new Error('update rejected');
  const file = join(await mkdtemp(join(tmpdir(), 'survng-update-')), 'archive.tgz');
  await pipeline(Readable.fromWeb(response.body as never), createWriteStream(file, { mode: 0o600 }));
  return file;
}

async function tarNames(file: string): Promise<string[]> {
  const child = spawn('tar', ['-tzf', file], { stdio: ['ignore', 'pipe', 'ignore'] });
  const chunks: Buffer[] = [];
  child.stdout.on('data', (chunk: Buffer) => chunks.push(chunk));
  const [code] = await once(child, 'close');
  if (code !== 0) throw new Error('update rejected');
  return Buffer.concat(chunks).toString('utf8').split('\n').map(name => name.trim()).filter(Boolean);
}

function assertArchive(names: string[], allow: (name: string) => boolean): void {
  if (!names.length || names.some(name => name.startsWith('/') || name.includes('\\') || name.split('/').includes('..') || !allow(name))) {
    throw new Error('update rejected');
  }
}

async function extractArchive(file: string, allow: (name: string) => boolean): Promise<string> {
  assertArchive(await tarNames(file), allow);
  // PrivateTmp makes /tmp a different filesystem from the install, so a rename into place fails.
  const directory = await mkdtemp(join(home, '.update-'));
  const child = spawn('tar', ['-xzf', file, '-C', directory, '--no-same-owner'], { stdio: 'ignore' });
  const [code] = await once(child, 'close');
  if (code !== 0) throw new Error('update rejected');
  return directory;
}

async function replacePath(name: string, source: string): Promise<void> {
  const destination = join(home, name);
  const backup = `${destination}.previous`;
  await rm(backup, { recursive: true, force: true });
  if (existsSync(destination)) await rename(destination, backup);
  try {
    await rename(source, destination);
  } catch (error) {
    if (existsSync(backup)) await rename(backup, destination);
    throw error;
  }
  await rm(backup, { recursive: true, force: true });
}

async function installFfmpeg(): Promise<void> {
  const file = await pullArchive('/api/homekit/nodes/ffmpeg');
  try {
    const directory = await extractArchive(file, name => name === 'ffmpeg' || name === 'ffprobe');
    await mkdir(join(home, 'bin'), { recursive: true });
    for (const name of ['ffmpeg', 'ffprobe']) {
      const destination = join(home, 'bin', name);
      await rename(join(directory, name), destination);
      await chmod(destination, 0o755);
    }
    await rm(directory, { recursive: true, force: true });
  } finally {
    await rm(dirname(file), { recursive: true, force: true });
  }
}

async function installAgent(): Promise<void> {
  const file = await pullArchive('/api/homekit/nodes/agent');
  try {
    const allow = (name: string) => name === 'package.json' || name === 'dist' || name.startsWith('dist/') || name === 'node_modules' || name.startsWith('node_modules/');
    const directory = await extractArchive(file, allow);
    await replacePath('dist', join(directory, 'dist'));
    await replacePath('node_modules', join(directory, 'node_modules'));
    await replacePath('package.json', join(directory, 'package.json'));
    await rm(directory, { recursive: true, force: true });
  } finally {
    await rm(dirname(file), { recursive: true, force: true });
  }
}

async function installRuntime(): Promise<void> {
  const file = await pullArchive('/api/homekit/nodes/runtime');
  try {
    const allow = (name: string) => /^(?:dri|dri\/iHD_drv_video\.so|lib[A-Za-z0-9._+-]+\.so(?:\.[A-Za-z0-9._+-]+)?)$/.test(name);
    const names = await tarNames(file);
    assertArchive(names, allow);
    if (!names.includes('libva.so.2') || !names.includes('libva-drm.so.2') || !names.includes('dri/iHD_drv_video.so')) throw new Error('update rejected');
    const directory = await extractArchive(file, allow);
    await replacePath('lib', directory);
  } finally {
    await rm(dirname(file), { recursive: true, force: true });
  }
}

async function applyUpdates(releases: Releases): Promise<void> {
  if (Date.now() < updateRetryAt) return;
  const local = await savedReleases();
  const ffmpegChanged = !!releases.ffmpeg && releases.ffmpeg !== local.ffmpeg;
  const agentChanged = !!releases.agent && releases.agent !== local.agent;
  const runtimeChanged = !!releases.runtime && releases.runtime !== local.runtime;
  if (!ffmpegChanged && !agentChanged && !runtimeChanged) return;
  try {
    if (agentChanged) await installAgent();
    if (ffmpegChanged) await installFfmpeg();
    if (runtimeChanged) await installRuntime();
    await writeFile(releasesPath, JSON.stringify({
      agent: releases.agent || local.agent || '', ffmpeg: releases.ffmpeg || local.ffmpeg || '', runtime: releases.runtime || local.runtime || '',
    }), { mode: 0o600 });
    for (const key of [...feeds.keys()]) stopFeed(key);
    process.exit(1);
  } catch {
    updateRetryAt = Date.now() + 60_000;
    console.error('SurvNG HomeKit encoder update failed.');
  }
}

async function prepare(camera: Assignment): Promise<void> {
  if (!profiles.has(camera.id)) {
    try {
      const frame = await probeFrame(ffmpeg, camera.url, camera.transport === 'udp' ? 'udp' : 'tcp');
      profiles.set(camera.id, streamProfile(frame.width, frame.height));
    } catch { return; }
  }
  if (encodings.has(camera.id)) return;
  const allow = !!camera.allowSoftwareEncoding;
  const directory = tmpdir();
  const admission = async () => {};
  try {
    encodings.set(camera.id, await selectEncoding(allow,
      async () => devicePresent() === 'qsv' && await qualifyEncoding(ffmpeg, directory, admission, AbortSignal.timeout(20000), { backend: 'qsv', device: '/dev/dri/renderD128' }),
      () => qualifyEncoding(ffmpeg, directory, admission, AbortSignal.timeout(20000), { backend: 'software' })));
  } catch (error) {
    if (error instanceof EncodingUnavailableError) encodings.set(camera.id, 'unavailable');
  }
}

async function main(): Promise<void> {
  let ffmpegVersion = '';
  const version = spawn(ffmpeg, ['-version'], { stdio: ['ignore', 'pipe', 'ignore'] });
  const banner: Buffer[] = [];
  version.stdout!.on('data', (chunk: Buffer) => banner.push(chunk));
  await new Promise(resolve => version.once('close', resolve));
  ffmpegVersion = Buffer.concat(banner).toString('utf8').split('\n')[0]?.slice(0, 80) || '';
  const stopAll = () => { for (const key of [...feeds.keys()]) stopFeed(key); };
  process.once('SIGTERM', () => { stopAll(); process.exit(0); });
  process.once('SIGINT', () => { stopAll(); process.exit(0); });
  for (;;) {
    const backends = Object.fromEntries([...encodings].map(([id, value]) => [id, value === 'unavailable' ? 'unavailable' : value.backend]));
    let desired: { assignments?: Assignment[]; feeds?: { camera: string; tier: number }[]; snapshots?: string[]; releases?: Releases; ingest?: string } = {};
    try {
      const response = await request('/api/homekit/nodes/sync', JSON.stringify({
        name: hostname().slice(0, 64), hardware: devicePresent(), ffmpeg: ffmpegVersion,
        profiles: Object.fromEntries(profiles), backends, fps: Object.fromEntries(rates),
      }), { 'content-type': 'application/json' });
      if (!response.ok) throw new Error('sync rejected');
      desired = await response.json() as typeof desired & { releases?: Releases };
    } catch {
      await new Promise(resolve => setTimeout(resolve, 3000));
      continue;
    }
    const ingest = typeof desired.ingest === 'string' ? desired.ingest.replace(/\/$/, '') : '';
    if (/^https?:\/\/[A-Za-z0-9.-]+(?::\d+)?(?:\/[A-Za-z0-9._~-]+)*$/.test(ingest)) mediaOrigin = ingest;
    await applyUpdates(desired.releases || {});
    const assignments = desired.assignments || [];
    const assigned = new Set(assignments.map(item => item.id));
    for (const id of [...profiles.keys()]) {
      if (!assigned.has(id)) { profiles.delete(id); encodings.delete(id); rates.delete(id); }
    }
    for (const camera of assignments) await prepare(camera);
    const wanted = new Set((desired.feeds || []).map(item => `${item.camera}:${item.tier}`));
    for (const key of [...feeds.keys()]) if (!wanted.has(key)) stopFeed(key);
    for (const item of desired.feeds || []) {
      const camera = assignments.find(candidate => candidate.id === item.camera);
      const encoding = camera && encodings.get(camera.id);
      const tier = camera && profiles.get(camera.id)?.tiers.find(candidate => candidate.identifier === item.tier);
      if (camera && tier && encoding && encoding !== 'unavailable') startFeed(camera, tier, encoding);
    }
    for (const id of desired.snapshots || []) {
      const camera = assignments.find(candidate => candidate.id === id);
      if (camera) await takeSnapshot(camera);
    }
  }
}

main().catch(() => { console.error('SurvNG HomeKit encoder stopped.'); process.exitCode = 1; });
