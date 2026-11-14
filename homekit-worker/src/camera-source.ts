import { createServer, type Socket } from 'node:net';
import { chmod } from 'node:fs/promises';
import { once } from 'node:events';
import type { ChildProcess } from 'node:child_process';
import { stopChild } from './media.js';

/** One normalized HEVC/AAC producer per requested tier, shared by consumers.
 * MPEG-TS allows each consumer to join at the next repeated parameter sets/IDR
 * without retaining camera media on disk or delaying all viewers for a slow one.
 */
export class CameraSource {
  private server = createServer(client => {
    if (this.closed) { client.destroy(); return; }
    this.clients.add(client);
    client.on('close', () => this.clients.delete(client));
    client.on('error', () => client.destroy());
    void (this.producing ??= this.produce()).catch(() => { void this.stop(); });
  });
  private clients = new Set<Socket>();
  private child?: ChildProcess;
  private starting?: Promise<string>;
  private producing?: Promise<void>;
  private stopping?: Promise<void>;
  private watchdog?: NodeJS.Timeout;
  closed = false;
  references = 0;

  constructor(private path: string, private producer: () => Promise<ChildProcess>) {}
  start(): Promise<string> { return this.starting ??= this.open(); }
  private async open(): Promise<string> {
    this.server.listen(this.path); await once(this.server, 'listening');
    await chmod(this.path, 0o600);
    return `unix://${this.path}`;
  }
  private async produce(): Promise<void> {
    try {
      this.child = await this.producer();
      if (this.closed) { await stopChild(this.child); return; }
      let lastData = Date.now();
      this.child.stdout!.on('data', (data: Buffer) => {
        lastData = Date.now();
        for (const client of this.clients) {
          // Bound each slow consumer independently; never block the producer.
          if (client.writableLength > 2 * 1024 * 1024) client.destroy();
          else client.write(data);
        }
      });
      this.child.on('error', () => { void this.stop(); });
      this.child.on('close', () => { void this.stop(); });
      this.watchdog = setInterval(() => { if (Date.now() - lastData > 15000) void this.stop(); }, 1000);
    } catch (error) { this.closed = true; throw error; }
  }
  stop(): Promise<void> {
    this.closed = true;
    return this.stopping ??= (async () => {
      clearInterval(this.watchdog);
      for (const client of this.clients) client.destroy();
      await this.producing?.catch(() => {});
      if (this.child) await stopChild(this.child);
      await new Promise<void>((resolve, reject) => this.server.close(error => {
        if (error && (error as NodeJS.ErrnoException).code !== 'ERR_SERVER_NOT_RUNNING') reject(error); else resolve();
      }));
    })();
  }
}
