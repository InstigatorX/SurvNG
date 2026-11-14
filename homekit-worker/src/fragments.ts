import { EventEmitter, once } from 'node:events';
import type { ChildProcess } from 'node:child_process';
import type { CameraRecordingConfiguration, CameraRecordingDelegate, RecordingPacket } from '@homebridge/hap-nodejs';
import { type MediaSource, tiers, stopChild, type Tier } from './media.js';

const MAX_BYTES = 64 * 1024 * 1024;
export function boxes(data: Buffer): Array<{ type: string; data: Buffer }> {
  const result = [];
  for (let offset = 0; offset < data.length;) {
    if (data.length - offset < 8) throw new Error('truncated MP4 box');
    const size = data.readUInt32BE(offset);
    if (size < 8 || size > data.length - offset) throw new Error('invalid MP4 box size');
    result.push({ type: data.toString('ascii', offset + 4, offset + 8), data: data.subarray(offset + 8, offset + size) });
    offset += size;
  }
  return result;
}

export function videoClock(init: Buffer): { track: number; scale: number } {
  const moov = boxes(init).find(box => box.type === 'moov');
  if (!moov) throw new Error('missing moov');
  for (const trak of boxes(moov.data).filter(box => box.type === 'trak')) {
    const contents = boxes(trak.data);
    const tkhd = contents.find(box => box.type === 'tkhd')!.data;
    const mdia = boxes(contents.find(box => box.type === 'mdia')!.data);
    const hdlr = mdia.find(box => box.type === 'hdlr')!.data;
    if (hdlr.toString('ascii', 8, 12) !== 'vide') continue;
    const mdhd = mdia.find(box => box.type === 'mdhd')!.data;
    const scale = mdhd.readUInt32BE(mdhd[0] === 1 ? 20 : 12);
    if (!scale) throw new Error('invalid video timescale');
    return { track: tkhd.readUInt32BE(tkhd[0] === 1 ? 20 : 12), scale };
  }
  throw new Error('missing video track');
}

export function decodeTime(moof: Buffer, track: number): bigint {
  const root = boxes(moof).find(box => box.type === 'moof');
  if (!root) throw new Error('missing moof');
  for (const traf of boxes(root.data).filter(box => box.type === 'traf')) {
    const contents = boxes(traf.data);
    const tfhd = contents.find(box => box.type === 'tfhd')!.data;
    if (tfhd.readUInt32BE(4) !== track) continue;
    const tfdt = contents.find(box => box.type === 'tfdt')!.data;
    return tfdt[0] === 1 ? tfdt.readBigUInt64BE(4) : BigInt(tfdt.readUInt32BE(4));
  }
  throw new Error('missing video decode time');
}

export function producerReferenceTime(track: number, mediaTime: bigint, wallMs: number): Buffer {
  const milliseconds = BigInt(Math.round(wallMs));
  const ntp = ((milliseconds / 1000n + 2208988800n) << 32n) | ((milliseconds % 1000n) << 32n) / 1000n;
  const box = Buffer.alloc(32);
  box.writeUInt32BE(32); box.write('prft', 4); box[8] = 1;
  box.writeUInt32BE(track, 12); box.writeBigUInt64BE(ntp, 16); box.writeBigUInt64BE(mediaTime, 24);
  return box;
}

export function fragmentDuration(moof: Buffer, track: number, scale: number): number {
  const root = boxes(moof).find(box => box.type === 'moof');
  if (!root) throw new Error('missing moof');
  for (const traf of boxes(root.data).filter(box => box.type === 'traf')) {
    const items = boxes(traf.data);
    const tfhd = items.find(box => box.type === 'tfhd')!.data;
    if (tfhd.readUInt32BE(4) !== track) continue;
    const flags = tfhd.readUInt32BE(0) & 0xffffff;
    let offset = 8 + (flags & 1 ? 8 : 0) + (flags & 2 ? 4 : 0);
    const defaultDuration = flags & 8 ? tfhd.readUInt32BE(offset) : 0;
    let total = 0;
    for (const trun of items.filter(box => box.type === 'trun')) {
      const data = trun.data, options = data.readUInt32BE(0) & 0xffffff, count = data.readUInt32BE(4);
      offset = 8 + (options & 1 ? 4 : 0) + (options & 4 ? 4 : 0);
      if (count > 10000) throw new Error('fragment sample limit exceeded');
      for (let i = 0; i < count; i++) {
        total += options & 0x100 ? data.readUInt32BE(offset) : defaultDuration;
        offset += (options & 0x100 ? 4 : 0) + (options & 0x200 ? 4 : 0) + (options & 0x400 ? 4 : 0) + (options & 0x800 ? 4 : 0);
      }
    }
    if (!total || !scale) throw new Error('missing fragment duration');
    return total / scale;
  }
  throw new Error('missing video track');
}

export interface Fragment { sequence: number; start: number; duration: number; data: Buffer }
/** A bounded lab prebuffer. Cursors never retain evicted fragments. */
export class FragmentRing {
  readonly changed = new EventEmitter();
  readonly fragments: Fragment[] = [];
  private sequence = 0;
  private bytes = 0;
  closed = false;
  constructor(private maxBytes = MAX_BYTES, private historySeconds = 12) {}
  append(data: Buffer, start: number, duration = 2): void {
    if (this.closed) throw new Error('fragment source closed');
    if (data.length > this.maxBytes) throw new Error('fragment exceeds buffer capacity');
    const last = this.fragments.at(-1);
    if (last && start <= last.start) throw new Error('media discontinuity');
    if (!Number.isFinite(duration) || duration <= 0 || duration > this.historySeconds) throw new Error('invalid fragment duration');
    if (last && start > last.start + last.duration + 0.1) throw new Error('media discontinuity');
    this.fragments.push({ sequence: ++this.sequence, start, duration, data }); this.bytes += data.length;
    while (this.bytes > this.maxBytes || this.fragments[0].start + this.fragments[0].duration <= start + duration - this.historySeconds) {
      this.bytes -= this.fragments.shift()!.data.length;
    }
    this.changed.emit('change');
  }
  get coverage(): number {
    const last = this.fragments.at(-1);
    return last ? last.start + last.duration - this.fragments[0].start : 0;
  }
  get size(): number { return this.bytes; }
  cursor(preRoll = 4): number {
    if (this.coverage < preRoll) throw new Error('prebuffer not ready');
    const last = this.fragments.at(-1)!;
    const end = last.start + last.duration;
    return [...this.fragments].reverse().find(fragment => fragment.start <= end - preRoll + 0.001)!.sequence;
  }
  async read(sequence: number, signal: AbortSignal): Promise<Fragment> {
    while (true) {
      signal.throwIfAborted();
      if (this.closed) throw new Error('fragment source closed');
      if (this.fragments.length && sequence < this.fragments[0].sequence) throw new Error('consumer fell behind fragment retention');
      const found = this.fragments.find(fragment => fragment.sequence === sequence);
      if (found) return found;
      const timeout = AbortSignal.timeout(8000);
      await once(this.changed, 'change', { signal: AbortSignal.any([signal, timeout]) });
    }
  }
  close(): void { this.closed = true; this.fragments.length = 0; this.bytes = 0; this.changed.emit('change'); }
}

export class FragmentParser {
  private pending: Buffer = Buffer.alloc(0);
  private init: Buffer[] = [];
  private moof?: Buffer;
  constructor(private onInit: (init: Buffer) => void, private onFragment: (fragment: Buffer) => void) {}
  push(chunk: Buffer): void {
    if (this.pending.length + chunk.length > MAX_BYTES) throw new Error('MP4 parser capacity exceeded');
    this.pending = Buffer.concat([this.pending, chunk]);
    while (this.pending.length >= 8) {
      const size = this.pending.readUInt32BE(0);
      if (size < 8 || size > MAX_BYTES) throw new Error('invalid MP4 box size');
      if (this.pending.length < size) return;
      const box = this.pending.subarray(0, size);
      this.pending = this.pending.subarray(size);
      const kind = box.toString('ascii', 4, 8);
      if (kind === 'ftyp' || kind === 'moov') {
        this.init.push(box);
        if (kind === 'moov') { this.onInit(Buffer.concat(this.init)); this.init = []; }
      } else if (kind === 'moof') {
        if (this.moof) throw new Error('missing fragment media');
        this.moof = box;
      } else if (kind === 'mdat') {
        if (!this.moof) throw new Error('missing fragment metadata');
        this.onFragment(Buffer.concat([this.moof, box])); this.moof = undefined;
      }
    }
  }
}

export class LabRecording implements CameraRecordingDelegate {
  ring = new FragmentRing();
  private init?: Buffer;
  private child?: ChildProcess;
  private watchdog?: NodeJS.Timeout;
  private config?: CameraRecordingConfiguration;
  private active = false;
  private desired = false;
  private audio = true;
  private transition: Promise<void> = Promise.resolve();
  private stream?: { id: number; abort: AbortController };
  private motion = false;
  private motionStopped = 0;
  private producerKey = '';
  readonly counters = { streams: 0, completed: 0, failed: 0, producerFailures: 0 };
  constructor(private media: MediaSource, private allowed: () => boolean, private catalog: readonly Tier[] = tiers) {}
  setMotion(value: boolean): void { this.motion = value; if (!value) this.motionStopped = Date.now(); }
  updateRecordingConfiguration(config?: CameraRecordingConfiguration): void { this.config = config; this.reconcile(); }
  updateRecordingActive(active: boolean): void { this.desired = active; this.reconcile(); }
  setAudio(active: boolean): void { if (active !== this.audio) { this.audio = active; this.reconcile(); } }
  reconcile(): void {
    this.transition = this.transition.then(async () => {
      const key = this.desired && this.config && this.allowed() ? JSON.stringify([this.config, this.audio]) : '';
      if (this.active && key === this.producerKey) return;
      await this.stopProducer();
      if (key) { await this.startProducer(); this.producerKey = key; }
    }).catch(() => { this.active = false; this.counters.producerFailures++; console.error('Recording producer transition failed.'); });
  }
  private async startProducer(): Promise<void> {
    const resolution = this.config!.videoCodec.resolution;
    const tier = this.catalog.find(t => t.width === resolution[0] && t.height === resolution[1] && t.frameRate === resolution[2])
      ?? this.catalog.find(t => t.width === resolution[0] && t.height === resolution[1]);
    if (!tier) throw new Error('unsupported recording resolution');
    this.ring = new FragmentRing(); this.init = undefined; this.active = true;
    const wall = Date.now();
    let lastFragment = Date.now();
    let clock: { track: number; scale: number };
    const parser = new FragmentParser(init => { this.init = init; clock = videoClock(init); }, fragment => {
      lastFragment = Date.now();
      if (!clock) throw new Error('fragment before init');
      const time = decodeTime(fragment, clock.track);
      const start = Number(time) / clock.scale;
      this.ring.append(Buffer.concat([producerReferenceTime(clock.track, time, wall + start * 1000), fragment]), start,
        fragmentDuration(fragment, clock.track, clock.scale));
    });
    const child = await this.media.start([...this.media.recording(tier, this.audio), '-tag:v', 'hvc1',
      '-movflags', '+frag_keyframe+empty_moov+default_base_moof', '-f', 'mp4', 'pipe:1']);
    this.child = child;
    const fail = () => {
      if (this.child !== child) return;
      this.counters.producerFailures++; this.active = false; this.ring.close();
      clearInterval(this.watchdog);
      this.stream?.abort.abort(); child.kill('SIGTERM');
    };
    child.stdout!.on('data', (data: Buffer) => { try { parser.push(data); } catch { fail(); } });
    child.on('error', fail); child.on('close', fail);
    this.watchdog = setInterval(() => { if (Date.now() - lastFragment > 15000) fail(); }, 1000);
  }
  async *handleRecordingStreamRequest(id: number, signal?: AbortSignal): AsyncGenerator<RecordingPacket> {
    if (this.stream || !this.active || !this.init || !this.allowed()) throw new Error('recording unavailable');
    const ring = this.ring;
    let sequence = ring.cursor();
    const abort = new AbortController(); this.stream = { id, abort };
    const effective = AbortSignal.any([abort.signal, ...(signal ? [signal] : []), AbortSignal.timeout(300_000)]);
    this.counters.streams++;
    try {
      yield { data: this.init, isLast: false };
      while (true) {
        const fragment = await ring.read(sequence++, effective);
        const last = !this.motion && Date.now() - this.motionStopped >= 4000;
        yield { data: fragment.data, isLast: last };
        if (last) { this.counters.completed++; return; }
      }
    } catch (error) { if (!abort.signal.aborted && !signal?.aborted) this.counters.failed++; throw error; }
    finally { if (this.stream?.id === id) this.stream = undefined; }
  }
  closeRecordingStream(id: number): void { if (this.stream?.id === id) this.stream.abort.abort(); }
  acknowledgeStream(id: number): void { this.closeRecordingStream(id); }
  private async stopProducer(): Promise<void> {
    clearInterval(this.watchdog);
    this.active = false; this.stream?.abort.abort(); this.ring.close(); this.init = undefined;
    this.producerKey = '';
    const child = this.child; this.child = undefined;
    if (child) await stopChild(child);
  }
  async close(): Promise<void> { this.desired = false; await this.transition; await this.stopProducer(); }
  status(): object { return { active: this.active, ready: this.ring.coverage >= 4, bufferSeconds: this.ring.coverage, bufferBytes: this.ring.size, ...this.counters }; }
}
