import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { stopChild } from '../src/media.js';

for (const scenario of ['success', 'missing tools', 'wrong digest', 'manual retry'] as const) test(`agent HTTP update: ${scenario}`, { timeout: 15000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-agent-process-'));
  let child: ChildProcess | undefined;
  const server = createServer();
  try {
    await mkdir(join(root, 'bin'));
    await writeFile(join(root, 'bin', 'ffmpeg'), '#!/bin/sh\nprintf "ffmpeg version review\\n"\n', { mode: 0o755 });
    await mkdir(join(root, 'dist'));
    await writeFile(join(root, 'dist', 'old'), 'old program');
    await writeFile(join(root, 'package.json'), '{"name":"old"}');
    const oldRelease = '0'.repeat(64);
    await writeFile(join(root, '.releases.json'), JSON.stringify({ agent: oldRelease }));
    const stage = join(root, 'stage');
    await mkdir(join(stage, 'dist', 'src'), { recursive: true });
    await mkdir(join(stage, 'node_modules'));
    await writeFile(join(stage, 'dist', 'src', 'agent.js'), '// new agent');
    await writeFile(join(stage, 'node_modules', 'version'), 'new dependencies');
    await writeFile(join(stage, 'package.json'), '{"name":"new"}');
    const archive = join(root, 'agent.tgz');
    const tar = spawn('tar', ['-czf', archive, '-C', stage, 'dist', 'node_modules', 'package.json']);
    assert.equal((await once(tar, 'close'))[0], 0);
    const payload = await readFile(archive);
    const release = scenario === 'wrong digest' ? 'f'.repeat(64) : createHash('sha256').update(payload).digest('hex');
    const firstRequest = '1'.repeat(32);
    const secondRequest = '2'.repeat(32);
    let retryRequested = false;
    let downloads = 0;
    let lastReport: { runningVersion?: string; failedSyncRequest?: string } | undefined;
    server.on('request', async (request, response) => {
      assert.equal(request.headers.authorization, 'Bearer test-only-agent-secret');
      if (request.url === '/api/homekit/nodes/sync') {
        const body: Buffer[] = [];
        for await (const part of request) body.push(part);
        lastReport = JSON.parse(Buffer.concat(body).toString('utf8')) as NonNullable<typeof lastReport>;
        if (lastReport.failedSyncRequest === firstRequest) retryRequested = true;
        setTimeout(() => { response.setHeader('content-type', 'application/json'); response.end(JSON.stringify({
          assignments: [], feeds: [], snapshots: [],
          releases: { agent: release, ...(scenario === 'missing tools' ? { ffmpeg: 'new-tools' } : {}) },
          syncRequest: scenario === 'manual retry' ? (retryRequested ? secondRequest : firstRequest) : undefined,
        })); }, 30);
      } else if (request.url === `/api/homekit/nodes/agent?release=${release}`) {
        downloads++;
        if (scenario === 'manual retry' && downloads === 1) {
          response.statusCode = 503;
          response.end();
        } else response.end(payload);
      } else { response.statusCode = 503; response.end(); }
    });
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    const port = (server.address() as { port: number }).port;
    child = spawn(process.execPath, [fileURLToPath(new URL('../src/agent.js', import.meta.url))], {
      cwd: root, env: { ...process.env, SURVNG_ORIGIN: `http://127.0.0.1:${port}`, SURVNG_NODE_SECRET: 'test-only-agent-secret', DEBUG: '', NODE_DEBUG: '' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let errors = '';
    child.stderr!.on('data', chunk => { errors = (errors + chunk.toString()).slice(-4096); });
    if (scenario === 'missing tools' || scenario === 'wrong digest') {
      const deadline = Date.now() + 10000;
      while (!errors.includes('encoder update failed')) {
        if (Date.now() > deadline || child.exitCode !== null) throw new Error(`agent failed before exercising rollback: ${errors}`);
        await new Promise(resolve => setTimeout(resolve, 20));
      }
      assert.equal(await readFile(join(root, 'dist', 'old'), 'utf8'), 'old program');
      assert.equal(await readFile(join(root, 'package.json'), 'utf8'), '{"name":"old"}');
      assert.equal(JSON.parse(await readFile(join(root, '.releases.json'), 'utf8')).agent, oldRelease);
      await stopChild(child);
    } else {
      const [code] = await once(child, 'close');
      assert.equal(code, 1, errors);
      assert.equal(await readFile(join(root, 'dist', 'src', 'agent.js'), 'utf8'), '// new agent');
      assert.equal(await readFile(join(root, 'node_modules', 'version'), 'utf8'), 'new dependencies');
      assert.equal(JSON.parse(await readFile(join(root, '.releases.json'), 'utf8')).agent, release);
    }
    assert.equal(lastReport?.runningVersion, oldRelease);
    assert.equal(downloads, scenario === 'manual retry' ? 2 : 1);
    assert.equal(retryRequested, scenario === 'manual retry');
  } finally {
    if (child) await stopChild(child);
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    await rm(root, { recursive: true, force: true });
  }
});
