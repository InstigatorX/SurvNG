import { spawn, type ChildProcess } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { once } from 'node:events';
import type { Readable } from 'node:stream';
import { encodeVideo, deviceArgs, type VideoEncoding } from './encoding.js';
import { CameraSource } from './camera-source.js';
import { tiers, stopChild, childStopRequested, type MediaSource, type Tier } from './media.js';

export type MediaAdmission = (method: 'acquire' | 'release' | 'started', token: string, detail?: { purpose?: 'encoder' | 'relay' | 'snapshot'; pid?: number }) => Promise<void>;

/** Real inputs are supplied over the supervisor pipe, never command arguments to Node. */
export class CameraMedia implements MediaSource {
  readonly children = new Set<ChildProcess>();
  private pending = new Set<Promise<ChildProcess>>();
  private closed = false;
  private sequence = 0;
  private sources = new Map<number, CameraSource>();
  private snapshots = new Set<ChildProcess>();
  lastError?: string;
  private progress = new Map<ChildProcess, { pid?: number; frame?: number; fps?: number; speed?: number; outTimeUs?: number; dupFrames?: number; dropFrames?: number }>();
  get encoderStatus(): object { return { backend: this.encoding.backend, fallbackReason: this.encoding.fallbackReason, sources: [...this.progress.values()] }; }
  readonly counters = { failures: 0, warningChunks: 0, admissionFailures: 0 };
  constructor(private url: string, private audio: boolean, private ffmpeg: string,
    private transport: 'tcp' | 'udp', private admission: MediaAdmission, private allowed: () => boolean = () => true, private directory = tmpdir(), readonly encoding: VideoEncoding = { backend: 'software' }) {}

  input(tier: Tier): string[] { return ['-rw_timeout', '15000000', '-f', 'mpegts', '-i', `survng-homekit:${tier.identifier}`]; }
  protected directInput(): string[] {
    // RTSP is a demuxer with its own socket timeout; rw_timeout applies to
    // protocol inputs such as HTTP and is rejected by the RTSP demuxer.
    return [...(/^rtsps?:/i.test(this.url)
      ? ['-rtsp_transport', this.transport, '-timeout', '15000000']
      : ['-rw_timeout', '15000000']), '-i', this.url,
      ...(!this.audio ? ['-re', '-f', 'lavfi', '-i', 'anullsrc=r=32000:cl=mono'] : [])];
  }
  video(_tier: Tier): string[] { return ['-c:v', 'copy']; }
  recording(tier: Tier, audio: boolean): string[] {
    return [...this.input(tier), '-map', '0:v:0', '-c:v', 'copy',
      ...(audio ? ['-map', '0:a:0', '-c:a', 'copy', '-bsf:a', 'aac_adtstoasc'] : ['-an'])];
  }
  private async acquireSource(tier: Tier): Promise<{ source: CameraSource; url: string }> {
    let source = this.sources.get(tier.identifier);
    if (!source || source.closed) {
      source = new CameraSource(join(this.directory, `source-${randomUUID()}.sock`), () => this.spawnAdmitted([...deviceArgs(this.encoding), ...this.directInput(), '-map', '0:v:0', ...encodeVideo(tier, this.encoding),
        '-map', this.audio ? '0:a:0' : '1:a:0', '-c:a', 'aac', '-ac', '1', '-ar', '32000', '-b:a', '32k',
        '-mpegts_flags', '+resend_headers', '-muxdelay', '0', '-f', 'mpegts', 'pipe:1'], 'encoder'));
      this.sources.set(tier.identifier, source);
    }
    source.references++;
    try { return { source, url: await source.start() }; }
    catch (error) { await this.releaseSource(tier, source); throw error; }
  }
  private async releaseSource(tier: Tier, source: CameraSource): Promise<void> {
    if (--source.references === 0) {
      if (this.sources.get(tier.identifier) === source) this.sources.delete(tier.identifier);
      await source.stop();
    }
  }
  private async startAdmitted(args: string[]): Promise<ChildProcess> {
    const sourceIndex = args.findIndex(value => value.startsWith('survng-homekit:'));
    if (sourceIndex < 0) return this.spawnAdmitted(args, args.includes('mjpeg') ? 'snapshot' : 'relay');
    const tier = tiers.find(tier => `survng-homekit:${tier.identifier}` === args[sourceIndex]);
    if (!tier) throw new Error('unsupported camera tier');
    const { source, url } = await this.acquireSource(tier);
    try {
      const input = [...args]; input[sourceIndex] = url;
      const child = await this.spawnAdmitted(input, 'relay');
      child.once('close', () => { void this.releaseSource(tier, source).catch(() => { this.counters.failures++; }); });
      return child;
    } catch (error) { await this.releaseSource(tier, source); throw error; }
  }
  start(args: string[]): Promise<ChildProcess> {
    if (this.closed) return Promise.reject(new Error('camera media closed'));
    const task = this.startAdmitted(args);
    this.pending.add(task);
    void task.then(() => this.pending.delete(task), () => this.pending.delete(task));
    return task;
  }
  private async spawnAdmitted(args: string[], purpose: 'encoder' | 'relay' | 'snapshot'): Promise<ChildProcess> {
    const token = String(++this.sequence);
    try { await this.admission('acquire', token, { purpose }); }
    catch {
      this.counters.admissionFailures++;
      await this.admission('release', token).catch(() => {});
      throw new Error('HomeKit media capacity unavailable');
    }
    if (this.closed || !this.allowed()) { await this.admission('release', token); throw new Error('camera media unavailable'); }
    let child: ChildProcess;
    try {
      child = spawn(this.ffmpeg, ['-hide_banner', '-loglevel', 'warning', '-nostdin', ...(purpose === 'encoder' ? ['-progress', 'pipe:3', '-stats_period', '1'] : []), ...args], { stdio: purpose === 'encoder' ? ['ignore', 'pipe', 'pipe', 'pipe'] : ['ignore', 'pipe', 'pipe'] });
    } catch { await this.admission('release', token); throw new Error('camera encoder unavailable'); }
    this.children.add(child);
    if (purpose === 'encoder') {
      const stats = { pid: child.pid };
      this.progress.set(child, stats);
      let pending = '';
      const keys: Record<string, string> = { frame: 'frame', fps: 'fps', speed: 'speed', out_time_us: 'outTimeUs', dup_frames: 'dupFrames', drop_frames: 'dropFrames' };
      (child.stdio[3] as Readable).on('data', (chunk: Buffer) => {
        pending += chunk.toString('utf8');
        if (pending.length > 8192) { pending = ''; return; }
        let end: number;
        while ((end = pending.indexOf('\n')) >= 0) {
          const line = pending.slice(0, end); pending = pending.slice(end + 1);
          const separator = line.indexOf('=');
          const key = keys[line.slice(0, separator)];
          const value = Number(line.slice(separator + 1).replace(/x$/, ''));
          if (key && Number.isFinite(value)) Object.assign(stats, { [key]: value });
        }
      });
    }
    // Camera/HTTP errors may contain credentials: retain counters, never stderr.
    child.stderr!.on('data', (chunk: Buffer) => {
      this.counters.warningChunks++;
      if (childStopRequested(child)) return;
      const value = chunk.toString('utf8');
      if (/401|403|unauthori[sz]ed|authentication/i.test(value)) this.lastError = 'authentication_failed';
      else if (/timed out|connection refused|no route|error opening input/i.test(value)) this.lastError = 'input_unavailable';
      else if (/error|failed|invalid/i.test(value)) this.lastError = 'media_error';
    });
    child.on('error', () => this.counters.failures++);
    child.once('close', (code, signal) => {
      this.children.delete(child);
      this.progress.delete(child);
      if ((code !== 0 || signal !== null) && !childStopRequested(child)) this.counters.failures++;
      void this.admission('release', token).catch(() => { this.counters.failures++; });
    });
    if (child.pid) {
      try { await this.admission('started', token, { pid: child.pid }); }
      catch { await stopChild(child); throw new Error('media process ownership unavailable'); }
    }
    return child;
  }
  async snapshot(): Promise<Buffer> {
    const child = await this.start([...this.directInput(), '-map', '0:v:0', '-frames:v', '1', '-an', '-vf', 'scale=1280:-2', '-c:v', 'mjpeg', '-f', 'image2pipe', 'pipe:1']);
    this.snapshots.add(child);
    const chunks: Buffer[] = []; let size = 0;
    const timer = setTimeout(() => child.kill('SIGKILL'), 10000);
    child.stdout!.on('data', (chunk: Buffer) => { size += chunk.length; if (size > 4 * 1024 * 1024) child.kill('SIGKILL'); else chunks.push(chunk); });
    try {
      const [code] = await once(child, 'close');
      if (code !== 0 || !size || size > 4 * 1024 * 1024) throw new Error('snapshot unavailable');
      return Buffer.concat(chunks);
    } finally { clearTimeout(timer); this.snapshots.delete(child); }
  }
  async cancelSnapshots(): Promise<void> { await Promise.all([...this.snapshots].map(stopChild)); }
  async stop(): Promise<void> {
    this.closed = true;
    await Promise.allSettled([...this.pending]);
    await Promise.all([...this.sources.values()].map(source => source.stop()));
    await Promise.all([...this.children].map(stopChild));
  }
}
