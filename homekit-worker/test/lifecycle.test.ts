import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtemp, readFile, lstat, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { connect, createServer } from 'node:net';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';

test('lab starts, preserves identity on restart, and removes runtime endpoints on shutdown', {
  skip: process.env.HKSV_MEDIA_TESTS !== '1', timeout: 90000,
}, async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-hksv-lifecycle-'));
  const portReservation = createServer(); portReservation.listen(0, '127.0.0.1'); await once(portReservation, 'listening');
  const port = (portReservation.address() as { port: number }).port;
  await new Promise<void>((resolve, reject) => portReservation.close(error => error ? reject(error) : resolve()));
  let child: ChildProcess | undefined;
  const request = async (method: string): Promise<{ result: Record<string, unknown> }> => {
    const socket = connect(join(root, 'control.sock')); socket.on('error', () => {}); await once(socket, 'connect');
    const chunks: Buffer[] = []; socket.on('data', data => chunks.push(data));
    socket.write(JSON.stringify({ version: 1, id: 'lifecycle', method }) + '\n');
    await once(socket, 'end'); return JSON.parse(Buffer.concat(chunks).toString());
  };
  try {
    let identity: string | undefined;
    for (let iteration = 0; iteration < 2; iteration++) {
      let diagnostics = '';
      child = spawn(process.execPath, [fileURLToPath(new URL('../src/lab.js', import.meta.url)),
        '--state-dir', root, '--bind', '127.0.0.1', '--port', String(port), '--allow-synthetic-encoding'],
      { stdio: ['ignore', 'pipe', 'pipe'] });
      const stopped = once(child, 'close');
      child.stdout!.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-4096); });
      child.stderr!.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-4096); });
      const deadline = Date.now() + 40000;
      while (!diagnostics.includes('lab ready')) {
        if (child.exitCode !== null) throw new Error(`lab exited: ${diagnostics}`);
        if (Date.now() > deadline) throw new Error(`lab startup timed out: ${diagnostics}`);
        await delay(50);
      }
      const status = (await request('status')).result;
      assert.equal(status.paired, false); assert.equal(status.mode, 'synthetic-lab');
      assert.ok(!JSON.stringify(status).includes('pincode'));
      assert.ok(String((await request('pairing')).result.setupUri).startsWith('X-HM://'));
      const current = await readFile(join(root, 'identity.json'), 'utf8');
      if (identity !== undefined) assert.equal(current, identity);
      identity = current;
      assert.equal((await lstat(join(root, 'identity.json'))).mode & 0o777, 0o600);
      child.kill('SIGTERM');
      const [code] = await stopped; assert.equal(code, 0, diagnostics); child = undefined;
      await assert.rejects(access(join(root, 'control.sock')));
      await assert.rejects(access(join(root, 'lab.lock')));
    }
  } finally {
    if (child && child.exitCode === null) {
      const stopped = once(child, 'close'); child.kill('SIGTERM');
      const timer = setTimeout(() => child?.kill('SIGKILL'), 5000);
      try { await stopped; } finally { clearTimeout(timer); }
    }
    await rm(root, { recursive: true, force: true });
  }
});
