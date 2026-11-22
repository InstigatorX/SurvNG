import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { createWriteStream, existsSync } from 'node:fs';
import { chmod, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { hostname, tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { encoderCommand } from './camera-media.js';
import { probeFrame, qualifyEncoding, selectEncoding, streamProfile, EncodingUnavailableError, type VideoEncoding } from './encoding.js';
import { stopChild, type Tier } from './media.js';
import { AgentFeeds, AgentJobs, changedAssignments } from './agent-state.js';
import { toolMembers, validateArchive, verifyEncoderTools, verifyToolRelease } from './agent-update.js';
import { activateGeneration, discardGeneration, preflightGeneration, stageGeneration } from './agent-generations.js';
import { AgentProgress, superviseAgent } from './agent-lifecycle.js';
import { mediaUploadOrigin, post } from './agent-http.js';

const origin = (process.env.SURVNG_ORIGIN || '').replace(/\/$/, '');
let mediaOrigin = origin;
const secret = process.env.SURVNG_NODE_SECRET || '';
const home = process.cwd();
const installRoot = process.env.SURVNG_ENCODER_HOME;
if (process.argv[2] === '--preflight') {
  try { await verifyEncoderTools(home); }
  catch { console.error('SurvNG HomeKit encoder preflight failed.'); process.exit(1); }
  process.exit(0);
}
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
  return post(`${base}${path}`, body, { authorization: `Bearer ${secret}`, ...headers }, AbortSignal.any(signals), stream);
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
    'x-survng-revision': camera.revision,
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
const releaseDigest = /^[0-9a-f]{64}$/;
const syncToken = /^[0-9a-f]{32}$/;
let updateRetryAt = 0;

async function savedReleases(): Promise<Releases> {
  try {
    const { agent, ffmpeg, runtime } = JSON.parse(await readFile(releasesPath, 'utf8')) as Releases;
    return { agent, ffmpeg, runtime };
  }
  catch { return {}; }
}

async function pullArchive(path: string, signal: AbortSignal, expectedDigest?: string): Promise<string> {
  const response = await fetch(`${origin}${path}`, { headers: { authorization: `Bearer ${secret}` },
    redirect: 'error', signal: AbortSignal.any([signal, AbortSignal.timeout(60000)]) });
  if (!response.ok || !response.body) throw new Error('update rejected');
  const file = join(await mkdtemp(join(tmpdir(), 'survng-update-')), 'archive.tgz');
  let size = 0;
  const digest = expectedDigest ? createHash('sha256') : undefined;
  const limit = new Transform({ transform(chunk: Buffer, _encoding, done) {
    size += chunk.length;
    if (size > 512 * 1024 * 1024) return done(new Error('update exceeds capacity'));
    digest?.update(chunk);
    done(null, chunk);
  } });
  try {
    await pipeline(Readable.fromWeb(response.body as never), limit, createWriteStream(file, { mode: 0o600 }));
    if (digest && digest.digest('hex') !== expectedDigest) throw new Error('update digest mismatch');
    return file;
  } catch (error) { await rm(dirname(file), { recursive: true, force: true }); throw error; }
}

type StagedUpdate = { directory: string; entries: { name: string; source: string }[] };
async function stageUpdate(kind: keyof Releases, release: string, signal: AbortSignal): Promise<StagedUpdate> {
  if (kind === 'agent' && (typeof release !== 'string' || !releaseDigest.test(release))) throw new Error('invalid agent release');
  const path = `/api/homekit/nodes/${kind}`;
  const file = kind === 'agent' ? await pullArchive(`${path}?release=${release}`, signal, release) : await pullArchive(path, signal);
  let directory: string | undefined;
  try {
    const allow = kind === 'agent'
      ? (name: string) => name === 'package.json' || /^dist(?:\/|$)/.test(name) || /^node_modules(?:\/|$)/.test(name)
      : (name: string) => toolMembers(kind).includes(name);
    const names = await validateArchive(file, allow, signal);
    const required = kind === 'agent' ? ['package.json', 'dist/src/agent.js']
      : toolMembers(kind);
    if (required.some(name => !names.includes(name))) throw new Error('update incomplete');
    // Keep staging on the install filesystem; PrivateTmp can be on another mount.
    directory = await mkdtemp(join(installRoot!, '.update-'));
    const child = spawn('tar', ['-xzf', file, '-C', directory, '--no-same-owner', '--no-same-permissions'],
      { stdio: 'ignore', timeout: 30000, killSignal: 'SIGKILL', signal });
    const [code] = await once(child, 'close');
    if (code !== 0) throw new Error('update rejected');
    if (kind !== 'agent') await verifyToolRelease(kind, release, directory, signal);
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

const components = ['agent', 'ffmpeg', 'runtime'] as const;

function releaseKey(releases: Releases): string {
  return JSON.stringify(components.map(kind => releases[kind] ?? ''));
}

function rejectedTrial(): (Releases & { syncRequest?: string }) | undefined {
  if (!process.env.SURVNG_ENCODER_REJECTED) return;
  try { return JSON.parse(process.env.SURVNG_ENCODER_REJECTED); }
  catch { console.error('SurvNG HomeKit could not read the rejected release record.'); }
}

async function prepareGeneration(root: string, releases: Releases, local: Releases, requestId: string | undefined,
  signal: AbortSignal): Promise<string> {
  const staged: StagedUpdate[] = [];
  const stampDirectory = await mkdtemp(join(root, '.update-stamp-'));
  let candidate: string | undefined;
  try {
    for (const kind of components) {
      const digest = releases[kind];
      if (digest && digest !== local[kind]) staged.push(await stageUpdate(kind, digest, signal));
    }
    const stamp = join(stampDirectory, '.releases.json');
    await writeFile(stamp, JSON.stringify({ ...releases, syncRequest: requestId }), { mode: 0o600 });
    signal.throwIfAborted();
    candidate = await stageGeneration(root, home, [...staged.flatMap(item => item.entries),
      { name: '.releases.json', source: stamp }]);
    await preflightGeneration(candidate, signal);
    signal.throwIfAborted();
    return candidate;
  } catch (error) {
    if (candidate) await discardGeneration(root, candidate);
    throw error;
  } finally {
    await Promise.all([...staged.map(item => item.directory), stampDirectory]
      .map(directory => rm(directory, { recursive: true, force: true }))).catch(() => {
        console.error('SurvNG HomeKit encoder staging cleanup failed.');
      });
  }
}

type UpdateJob = { key: string; requestId?: string; abort: AbortController; work?: Promise<void>;
  candidate?: string; done: boolean; failed: boolean };
let updating: UpdateJob | undefined;

async function cancelUpdate(): Promise<void> {
  const job = updating;
  if (!job) return;
  job.abort.abort();
  await job.work;
  if (job.candidate) {
    await discardGeneration(installRoot!, job.candidate).catch(() => {
      console.error('SurvNG HomeKit encoder candidate cleanup failed.');
    });
  }
  updating = undefined;
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
  // A heartbeat describes the running process, even if an installer has since
  // written a new release stamp to disk.
  const localReleases = await savedReleases();
  const runningVersion = typeof localReleases.agent === 'string' && releaseDigest.test(localReleases.agent) ? localReleases.agent : null;
  const rejected = rejectedTrial();
  let handledSyncRequest = rejected?.syncRequest;
  let failedSyncRequest = rejected?.syncRequest;
  let ffmpegVersion = '';
  const version = spawn(ffmpeg, ['-version'], { stdio: ['ignore', 'pipe', 'ignore'], signal: shutdown.signal,
    timeout: 5000, killSignal: 'SIGKILL' });
  const banner: Buffer[] = [];
  let bannerSize = 0;
  version.stdout!.on('data', (chunk: Buffer) => { if (bannerSize < 4096) banner.push(chunk.subarray(0, 4096 - bannerSize)); bannerSize += chunk.length; });
  const [versionCode] = await once(version, 'close');
  if (versionCode !== 0) throw new Error('FFmpeg unavailable');
  ffmpegVersion = Buffer.concat(banner).toString('utf8').split('\n')[0]?.slice(0, 80) || '';
  if (installRoot && process.send) {
    // The primary may mark Update Sync successful as soon as it sees the new
    // version. Wait for the launcher's durable confirmation before reporting.
    const confirmation = once(process, 'message', {
      signal: AbortSignal.any([shutdown.signal, AbortSignal.timeout(5000)]),
    });
    process.send('ready');
    if ((await confirmation)[0] !== 'confirmed') throw new Error('Encoder startup was not confirmed');
  }
  let preparationCursor = 0;
  let warnedAboutIngest = false;
  while (!shutdown.signal.aborted) {
    const backends = Object.fromEntries([...encodings].map(([id, value]) => [id, value === 'unavailable' ? 'unavailable' : value.backend]));
    let desired: { assignments?: Assignment[]; feeds?: { camera: string; tier: number }[]; snapshots?: string[];
      snapshotRequests?: Record<string, string>; releases?: Releases; ingest?: string | null; syncRequest?: string } = {};
    try {
      progress.enter('sync', 20000);
      const response = await request('/api/homekit/nodes/sync', JSON.stringify({
        name: hostname().slice(0, 64), hardware: devicePresent(), ffmpeg: ffmpegVersion,
        runningVersion, failedSyncRequest, releaseProtocol: installRoot && process.send ? 1 : undefined,
        profiles: Object.fromEntries(profiles), backends, fps: Object.fromEntries(rates), revisions: Object.fromEntries(revisions),
      }), { 'content-type': 'application/json' });
      if (!response.ok) throw new Error('sync rejected');
      progress.enter('sync body', 20000);
      desired = await response.json() as typeof desired & { releases?: Releases };
    } catch {
      updating?.abort.abort();
      progress.enter('sync cleanup', 15000);
      await Promise.all([feeds.stopAll(), preparations.stopAll(), snapshots.stopAll()]);
      if (shutdown.signal.aborted) break;
      progress.enter('retry', 10000);
      await new Promise(resolve => setTimeout(resolve, 3000));
      continue;
    }
    let nextMediaOrigin = origin;
    try {
      nextMediaOrigin = mediaUploadOrigin(origin, desired.ingest);
      warnedAboutIngest = false;
    } catch {
      if (!warnedAboutIngest) console.error('HomeKit rejected an invalid media endpoint. Using the enrolled server.');
      warnedAboutIngest = true;
    }
    if (nextMediaOrigin !== mediaOrigin) {
      progress.enter('assignment cleanup', 15000);
      await feeds.stopAll();
      mediaOrigin = nextMediaOrigin;
    }
    const requestId = typeof desired.syncRequest === 'string' && syncToken.test(desired.syncRequest)
      ? desired.syncRequest : undefined;
    let requested = requestId !== undefined && requestId !== handledSyncRequest;
    const advertised = desired.releases;
    if (installRoot) {
      const target = { ...localReleases };
      for (const kind of components) {
        const version = advertised?.[kind];
        if (typeof version === 'string' && releaseDigest.test(version)) target[kind] = version;
      }
      const key = releaseKey(target);
      const changes = key !== releaseKey(localReleases);
      const job = updating;
      if (job && (!changes || key !== job.key || (requested && requestId !== job.requestId))) job.abort.abort();
      if (job?.done) {
        if (job.abort.signal.aborted || job.failed) {
          await cancelUpdate();
          if (job.failed && job.requestId) failedSyncRequest = job.requestId;
          if (job.abort.signal.aborted && !job.failed && job.requestId === requestId && requestId !== undefined) {
            handledSyncRequest = undefined;
            requested = true;
          }
        } else if (job.candidate) {
          progress.enter('updates', 30000);
          try {
            await Promise.all([feeds.stopAll(), preparations.stopAll(), snapshots.stopAll()]);
            shutdown.signal.throwIfAborted();
            await activateGeneration(installRoot, job.candidate);
            updating = undefined;
            process.exitCode = 1;
            shutdown.abort();
            break;
          } catch {
            await cancelUpdate();
            if (shutdown.signal.aborted) break;
            if (job.requestId) failedSyncRequest = job.requestId;
            console.error('SurvNG HomeKit encoder activation failed.');
            updateRetryAt = Date.now() + 60_000;
          }
        }
      }
      if (!updating && changes && (requested || (Date.now() >= updateRetryAt && (!rejected || key !== releaseKey(rejected))))) {
        if (requested) { handledSyncRequest = requestId; failedSyncRequest = undefined; }
        const next: UpdateJob = { key, requestId: requested ? requestId : undefined, abort: new AbortController(), done: false, failed: false };
        updating = next;
        next.work = prepareGeneration(installRoot, target, localReleases, next.requestId, next.abort.signal)
          .then(candidate => { next.candidate = candidate; }, () => {
            if (!next.abort.signal.aborted) {
              next.failed = true;
              updateRetryAt = Date.now() + 60_000;
              console.error('SurvNG HomeKit encoder update failed.');
            }
          }).finally(() => { next.done = true; });
      }
    }
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
    shutdown.abort(); await Promise.all([feeds.stopAll(), preparations.stopAll(), snapshots.stopAll(), cancelUpdate()]);
  });
superviseAgent(running, shutdown, progress);
