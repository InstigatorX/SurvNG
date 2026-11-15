import type { ChildProcess } from 'node:child_process';
import { stopChild } from './media.js';

/** Owns each uploader through child closure. Late HTTP results belong to that child. */
export class AgentFeeds {
  private children = new Map<string, ChildProcess>();
  private stopping = new Map<ChildProcess, Promise<void>>();
  private uploads = new Map<ChildProcess, () => void>();
  has(key: string): boolean { return this.children.has(key); }
  keys(): string[] { return [...this.children.keys()]; }

  track(key: string, child: ChildProcess, upload: Promise<unknown>, cancelUpload: () => void = () => {}): void {
    if (this.children.has(key)) throw new Error('encoder feed already owned');
    this.children.set(key, child);
    this.uploads.set(child, cancelUpload);
    child.once('close', () => {
      this.cancelUpload(child);
      if (this.children.get(key) === child) this.children.delete(key);
    });
    const stop = () => { void this.stop(key, child).catch(() => { console.error('SurvNG HomeKit feed shutdown failed.'); }); };
    child.once('error', stop);
    // An HTTP rejection, an ended upload, and a network failure all end this producer.
    void upload.then(stop, stop);
  }

  async stop(key: string, expected = this.children.get(key)): Promise<void> {
    if (!expected || this.children.get(key) !== expected) return;
    let pending = this.stopping.get(expected);
    if (!pending) {
      this.cancelUpload(expected);
      pending = stopChild(expected).catch(() => {
        // A failed spawn also emits close. It has no process to terminate.
        if (expected.pid) throw new Error('encoder feed shutdown failed');
      }).finally(() => {
        if (this.children.get(key) === expected) this.children.delete(key);
        this.stopping.delete(expected);
      });
      this.stopping.set(expected, pending);
    }
    await pending;
  }

  private cancelUpload(child: ChildProcess): void {
    const cancel = this.uploads.get(child);
    this.uploads.delete(child);
    cancel?.();
  }

  async stopAll(): Promise<void> { await Promise.all(this.keys().map(key => this.stop(key))); }
}

/** Bounded background work must not hold up the node heartbeat or survive revocation. */
export class AgentJobs {
  private jobs = new Map<string, { abort: AbortController; done: Promise<void> }>();
  constructor(private limit: number) {}
  keys(): string[] { return [...this.jobs.keys()]; }
  start(key: string, run: (signal: AbortSignal) => Promise<void>, failed: () => void): boolean {
    if (this.jobs.has(key) || this.jobs.size >= this.limit) return false;
    const abort = new AbortController();
    const job = { abort, done: Promise.resolve() };
    job.done = Promise.resolve().then(() => { abort.signal.throwIfAborted(); return run(abort.signal); })
      .catch(() => { if (!abort.signal.aborted) failed(); })
      .finally(() => { if (this.jobs.get(key) === job) this.jobs.delete(key); });
    this.jobs.set(key, job);
    return true;
  }
  async stop(key: string): Promise<void> {
    const job = this.jobs.get(key);
    if (job) { job.abort.abort(); await job.done; }
  }
  async stopAll(): Promise<void> { await Promise.all(this.keys().map(key => this.stop(key))); }
}

export function changedAssignments<T extends { id: string }>(previous: Map<string, T>, assignments: T[]): string[] {
  const incoming = new Map(assignments.map(item => [item.id, item]));
  const changed = [...previous].filter(([id, before]) => JSON.stringify(incoming.get(id)) !== JSON.stringify(before)).map(([id]) => id);
  previous.clear();
  for (const [id, value] of incoming) previous.set(id, value);
  return changed;
}
