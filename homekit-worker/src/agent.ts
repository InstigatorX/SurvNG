import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createWriteStream, existsSync } from 'node:fs';
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { hostname, tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { encoderCommand } from './camera-media.js';
import { probeFrame, qualifyEncoding, selectEncoding, streamProfile, EncodingUnavailableError, type VideoEncoding } from './encoding.js';
import { stopChild, type Tier } from './media.js';
import { AgentFeeds, AgentJobs, changedAssignments } from './agent-state.js';
import { replaceUpdate, validateArchive } from './agent-update.js';
import { AgentProgress, superviseAgent } from './agent-lifecycle.js';

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

type Assignment = { id: string; url: string; audio: boolean; transport: 'tcp' | 'udp'; allowSoftwareEncoding: boolean; revision: string };
type Profile = ReturnType<typeof streamProfile>;
const profiles = new Map<string, Profile>();
const encodings = new Map<string, VideoEncoding | 'unavailable'>();
const rates = new Map<string, number>();
const revisions = new Map<string, string>();
const feeds = new AgentFeeds();
const previousAssignments = new Map<string, Assignment>();
const preparations = new AgentJobs(1);
const snapshots = new AgentJobs(2);
const snapshotRequests = new Map<string, string>();
const shutdown = new AbortController();
const progress = new AgentProgress();

function devicePresent(): 'qsv' | 'software' {
  return existsSync('/dev/dri/renderD128') ? 'qsv' : 'software';
}

async function request(path: string, body: BodyInit | ReadableStream | NodeJS.ReadableStream, headers: Record<string, string>, stream = false, base = origin, signal?: AbortSignal): Promise<Response> {
  const signals = [shutdown.signal, ...(signal ? [signal] : []), ...(!stream ? [AbortSignal.timeout(15000)] : [])];
  return fetch(`${base}${path}`, { method: 'POST', headers: { authorization: `Bearer ${secret}`, ...headers }, body: body as BodyInit,
    signal: AbortSignal.any(signals),
    ...(stream ? { duplex: 'half' } : {}) } as RequestInit);
}

function startFeed(camera: Assignment, tier: Tier, encoding: VideoEncoding): void {
  const key = `${camera.id}:${tier.identifier}`;
  if (feeds.has(key)) return;
  const child = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-nostdin', '-progress', 'pipe:3', '-stats_period', '1',
    ...encoderCommand(camera.url, camera.transport, camera.audio, tier, encoding)], { stdio: ['ignore', 'pipe', 'ignore', 'pipe'] });
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
  const upload = new AbortController();
  feeds.track(key, child, request('/api/homekit/nodes/media', child.stdout!, {
    'content-type': 'application/octet-stream', 'x-survng-camera': camera.id, 'x-survng-tier': String(tier.identifier),
  }, true, mediaOrigin, upload.signal), () => upload.abort());
}

async function takeSnapshot(camera: Assignment, requestId: string, signal: AbortSignal): Promise<void> {
  signal.throwIfAborted();
  const child = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-nostdin',
    ...( /^rtsps?:/i.test(camera.url) ? ['-rtsp_transport', camera.transport, '-timeout', '15000000'] : ['-rw_timeout', '15000000']),
    '-i', camera.url, '-map', '0:v:0', '-frames:v', '1', '-an', '-vf', 'scale=1280:-2', '-c:v', 'mjpeg', '-f', 'image2pipe', 'pipe:1'],
    { stdio: ['ignore', 'pipe', 'ignore'], signal });
  const chunks: Buffer[] = [];
  let size = 0;
  child.stdout!.on('data', (chunk: Buffer) => {
    size += chunk.length;
    if (size > 4 * 1024 * 1024) child.kill('SIGKILL');
    else chunks.push(chunk);
  });
  const timer = setTimeout(() => child.kill('SIGKILL'), 10000);
  try {
    const [code] = await once(child, 'close');
    signal.throwIfAborted();
    if (code !== 0 || size === 0 || size > 4 * 1024 * 1024) return;
    const response = await request('/api/homekit/nodes/snapshot', Buffer.concat(chunks), {
      'content-type': 'image/jpeg', 'x-survng-camera': camera.id, 'x-survng-request': requestId }, false, origin, signal);
    if (!response.ok && response.status !== 404) throw new Error('snapshot upload failed');
  } finally { clearTimeout(timer); await stopChild(child); }
}

type Releases = { agent?: string; ffmpeg?: string; runtime?: string };
const releasesPath = join(home, '.releases.json');
let updateRetryAt = 0;

async function savedReleases(): Promise<Releases> {
  try { return JSON.parse(await readFile(releasesPath, 'utf8')) as Releases; }
  catch { return {}; }
}

async function pullArchive(path: string): Promise<string> {
  const response = await fetch(`${origin}${path}`, { headers: { authorization: `Bearer ${secret}` },
    signal: AbortSignal.any([shutdown.signal, AbortSignal.timeout(60000)]) });
  if (!response.ok || !response.body) throw new Error('update rejected');
  const file = join(await mkdtemp(join(tmpdir(), 'survng-update-')), 'archive.tgz');
  let size = 0;
  const limit = new Transform({ transform(chunk: Buffer, _encoding, done) {
    size += chunk.length;
    done(size > 512 * 1024 * 1024 ? new Error('update exceeds capacity') : null, chunk);
  } });
  try {
    await pipeline(Readable.fromWeb(response.body as never), limit, createWriteStream(file, { mode: 0o600 }));
    return file;
  } catch (error) { await rm(dirname(file), { recursive: true, force: true }); throw error; }
}

type StagedUpdate = { directory: string; entries: { name: string; source: string }[] };
async function stageUpdate(kind: 'agent' | 'ffmpeg' | 'runtime'): Promise<StagedUpdate> {
  const file = await pullArchive(`/api/homekit/nodes/${kind}`);
  let directory: string | undefined;
  try {
    const allow = kind === 'agent'
      ? (name: string) => name === 'package.json' || /^dist(?:\/|$)/.test(name) || /^node_modules(?:\/|$)/.test(name)
      : kind === 'ffmpeg' ? (name: string) => name === 'ffmpeg' || name === 'ffprobe'
      : (name: string) => /^(?:dri\/?|dri\/iHD_drv_video\.so|lib[A-Za-z0-9._+-]+\.so(?:\.[A-Za-z0-9._+-]+)?)$/.test(name);
    const names = await validateArchive(file, allow);
    const required = kind === 'agent' ? ['package.json', 'dist/src/agent.js']
      : kind === 'ffmpeg' ? ['ffmpeg', 'ffprobe'] : ['libva.so.2', 'libva-drm.so.2', 'dri/iHD_drv_video.so'];
    if (required.some(name => !names.includes(name))) throw new Error('update incomplete');
    // Keep staging on the install filesystem; PrivateTmp can be on another mount.
    directory = await mkdtemp(join(home, '.update-'));
    const child = spawn('tar', ['-xzf', file, '-C', directory, '--no-same-owner', '--no-same-permissions'],
      { stdio: 'ignore', timeout: 30000, killSignal: 'SIGKILL' });
    const [code] = await once(child, 'close');
    if (code !== 0) throw new Error('update rejected');
    if (kind === 'ffmpeg') {
      for (const name of ['ffmpeg', 'ffprobe']) await chmod(join(directory, name), 0o755);
    }
    const entries = kind === 'agent' ? ['dist', 'node_modules', 'package.json'].map(name => ({ name, source: join(directory!, name) }))
      : [{ name: kind === 'ffmpeg' ? 'bin' : 'lib', source: directory }];
    return { directory, entries };
  } catch (error) {
    if (directory) await rm(directory, { recursive: true, force: true });
    throw error;
  } finally { await rm(dirname(file), { recursive: true, force: true }); }
}

async function applyUpdates(releases: Releases): Promise<void> {
  if (Date.now() < updateRetryAt) return;
  const local = await savedReleases();
  const changed = (['agent', 'ffmpeg', 'runtime'] as const).filter(kind => releases[kind] && releases[kind] !== local[kind]);
  if (!changed.length) return;
  const staged: StagedUpdate[] = [];
  let stampDirectory: string | undefined;
  try {
    for (const kind of changed) staged.push(await stageUpdate(kind));
    stampDirectory = await mkdtemp(join(home, '.update-stamp-'));
    const stamp = join(stampDirectory, '.releases.json');
    await writeFile(stamp, JSON.stringify({ ...local, ...releases }), { mode: 0o600 });
    await Promise.all([feeds.stopAll(), preparations.stopAll(), snapshots.stopAll()]);
    await replaceUpdate(home, [...staged.flatMap(item => item.entries), { name: '.releases.json', source: stamp }]);
    // Exit only after the entire release has committed. systemd starts the new program.
    process.exitCode = 1;
    shutdown.abort();
  } catch {
    updateRetryAt = Date.now() + 60_000;
    console.error('SurvNG HomeKit encoder update failed.');
  } finally {
    for (const item of staged) await rm(item.directory, { recursive: true, force: true });
    if (stampDirectory) await rm(stampDirectory, { recursive: true, force: true });
  }
}

async function prepare(camera: Assignment, signal: AbortSignal): Promise<void> {
  if (encodings.has(camera.id)) return;
  const frame = await probeFrame(ffmpeg, camera.url, camera.transport === 'udp' ? 'udp' : 'tcp', signal);
  const profile = streamProfile(frame.width, frame.height);
  const allow = !!camera.allowSoftwareEncoding;
  const directory = tmpdir();
  const admission = async () => {};
  try {
    const encoding = await selectEncoding(allow,
      async () => devicePresent() === 'qsv' && await qualifyEncoding(ffmpeg, directory, admission, AbortSignal.any([signal, AbortSignal.timeout(20000)]), { backend: 'qsv', device: '/dev/dri/renderD128' }),
      () => qualifyEncoding(ffmpeg, directory, admission, AbortSignal.any([signal, AbortSignal.timeout(20000)]), { backend: 'software' }));
    signal.throwIfAborted();
    profiles.set(camera.id, profile);
    encodings.set(camera.id, encoding);
    revisions.set(camera.id, camera.revision);
  } catch (error) {
    signal.throwIfAborted();
    if (error instanceof EncodingUnavailableError) {
      profiles.set(camera.id, profile); encodings.set(camera.id, 'unavailable');
      revisions.set(camera.id, camera.revision);
    } else throw error;
  }
}

async function main(): Promise<void> {
  progress.enter('version', 10000);
  let ffmpegVersion = '';
  const version = spawn(ffmpeg, ['-version'], { stdio: ['ignore', 'pipe', 'ignore'], signal: shutdown.signal,
    timeout: 5000, killSignal: 'SIGKILL' });
  const banner: Buffer[] = [];
  let bannerSize = 0;
  version.stdout!.on('data', (chunk: Buffer) => { if (bannerSize < 4096) banner.push(chunk.subarray(0, 4096 - bannerSize)); bannerSize += chunk.length; });
  const [versionCode] = await once(version, 'close');
  if (versionCode !== 0) throw new Error('FFmpeg unavailable');
  ffmpegVersion = Buffer.concat(banner).toString('utf8').split('\n')[0]?.slice(0, 80) || '';
  let preparationCursor = 0;
  while (!shutdown.signal.aborted) {
    const backends = Object.fromEntries([...encodings].map(([id, value]) => [id, value === 'unavailable' ? 'unavailable' : value.backend]));
    let desired: { assignments?: Assignment[]; feeds?: { camera: string; tier: number }[]; snapshots?: string[];
      snapshotRequests?: Record<string, string>; releases?: Releases; ingest?: string } = {};
    try {
      progress.enter('sync', 20000);
      const response = await request('/api/homekit/nodes/sync', JSON.stringify({
        name: hostname().slice(0, 64), hardware: devicePresent(), ffmpeg: ffmpegVersion,
        profiles: Object.fromEntries(profiles), backends, fps: Object.fromEntries(rates), revisions: Object.fromEntries(revisions),
      }), { 'content-type': 'application/json' });
      if (!response.ok) throw new Error('sync rejected');
      progress.enter('sync body', 20000);
      desired = await response.json() as typeof desired & { releases?: Releases };
    } catch {
      progress.enter('sync cleanup', 15000);
      await Promise.all([feeds.stopAll(), preparations.stopAll(), snapshots.stopAll()]);
      if (shutdown.signal.aborted) break;
      progress.enter('retry', 10000);
      await new Promise(resolve => setTimeout(resolve, 3000));
      continue;
    }
    const ingest = typeof desired.ingest === 'string' ? desired.ingest.replace(/\/$/, '') : '';
    if (/^https?:\/\/[A-Za-z0-9.-]+(?::\d+)?(?:\/[A-Za-z0-9._~-]+)*$/.test(ingest)) mediaOrigin = ingest;
    progress.enter('updates', 300000);
    await applyUpdates(desired.releases || {});
    if (shutdown.signal.aborted) break;
    const assignments = desired.assignments || [];
    progress.enter('assignment cleanup', 30000);
    for (const id of changedAssignments(previousAssignments, assignments)) {
      await preparations.stop(id); await snapshots.stop(id); snapshotRequests.delete(id);
      await Promise.all(feeds.keys().filter(key => key.startsWith(`${id}:`)).map(key => feeds.stop(key)));
      profiles.delete(id); encodings.delete(id); rates.delete(id); revisions.delete(id);
    }
    const assigned = new Set(assignments.map(item => item.id));
    for (const id of [...profiles.keys()]) {
      if (!assigned.has(id)) { profiles.delete(id); encodings.delete(id); rates.delete(id); revisions.delete(id); }
    }
    const wanted = new Set((desired.feeds || []).map(item => `${item.camera}:${item.tier}`));
    progress.enter('feed cleanup', 30000);
    for (const key of feeds.keys()) if (!wanted.has(key)) await feeds.stop(key);
    for (let offset = 0; offset < assignments.length; offset++) {
      const index = (preparationCursor + offset) % assignments.length;
      const camera = assignments[index];
      if (!encodings.has(camera.id) && preparations.start(camera.id, signal => prepare(camera, signal),
        () => console.error('SurvNG HomeKit camera qualification failed.'))) {
        preparationCursor = (index + 1) % assignments.length;
        break;
      }
    }
    for (const item of desired.feeds || []) {
      const camera = assignments.find(candidate => candidate.id === item.camera);
      const encoding = camera && encodings.get(camera.id);
      const tier = camera && profiles.get(camera.id)?.tiers.find(candidate => candidate.identifier === item.tier);
      if (camera && tier && encoding && encoding !== 'unavailable') startFeed(camera, tier, encoding);
    }
    const wantedSnapshots = desired.snapshotRequests || {};
    progress.enter('snapshots', 15000);
    for (const id of snapshots.keys()) {
      if (snapshotRequests.get(id) !== wantedSnapshots[id]) await snapshots.stop(id);
    }
    for (const id of [...snapshotRequests.keys()]) if (!wantedSnapshots[id]) snapshotRequests.delete(id);
    for (const id of desired.snapshots || []) {
      const camera = assignments.find(candidate => candidate.id === id);
      const requestId = desired.snapshotRequests?.[id];
      if (camera && requestId && snapshotRequests.get(id) !== requestId) {
        if (snapshots.start(id, signal => takeSnapshot(camera, requestId, signal),
          () => console.error('SurvNG HomeKit snapshot failed.'))) snapshotRequests.set(id, requestId);
      }
    }
  }
}

const running = main().catch(() => { console.error('SurvNG HomeKit encoder stopped.'); process.exitCode = 1; })
  .finally(async () => {
    progress.enter('shutdown', 15000);
    shutdown.abort(); await Promise.all([feeds.stopAll(), preparations.stopAll(), snapshots.stopAll()]);
  });
superviseAgent(running, shutdown, progress);
