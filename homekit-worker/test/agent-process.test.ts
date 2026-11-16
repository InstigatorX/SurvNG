import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, writeFile, readFile, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { stopChild } from '../src/media.js';

for (const rejectTools of [false, true]) test(`agent HTTP update ${rejectTools ? 'failure preserves the installed release' : 'commits the entire staged release and exits'}`, { timeout: 15000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-agent-process-'));
  let child: ChildProcess | undefined;
  const server = createServer();
  try {
    await mkdir(join(root, 'bin'));
    await writeFile(join(root, 'bin', 'ffmpeg'), '#!/bin/sh\nprintf "ffmpeg version review\\n"\n', { mode: 0o755 });
    await mkdir(join(root, 'dist'));
    await writeFile(join(root, 'dist', 'old'), 'old program');
    await writeFile(join(root, 'package.json'), '{"name":"old"}');
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
    let uploads = 0;
    server.on('request', (request, response) => {
      assert.equal(request.headers.authorization, 'Bearer test-only-agent-secret');
      request.resume();
      if (request.url?.endsWith('/sync')) {
        setTimeout(() => { response.setHeader('content-type', 'application/json'); response.end(JSON.stringify({
          assignments: [], feeds: [], snapshots: [], releases: { agent: 'new-release', ...(rejectTools ? { ffmpeg: 'new-tools' } : {}) },
        })); }, 30);
      } else if (request.url?.endsWith('/agent')) { uploads++; response.end(payload); }
      else { response.statusCode = 503; response.end(); }
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
    if (rejectTools) {
      const deadline = Date.now() + 10000;
      while (!errors.includes('encoder update failed')) {
        if (Date.now() > deadline || child.exitCode !== null) throw new Error(`agent failed before exercising rollback: ${errors}`);
        await new Promise(resolve => setTimeout(resolve, 20));
      }
      assert.equal(await readFile(join(root, 'dist', 'old'), 'utf8'), 'old program');
      assert.equal(await readFile(join(root, 'package.json'), 'utf8'), '{"name":"old"}');
      await assert.rejects(access(join(root, '.releases.json')));
      await stopChild(child);
    } else {
      const [code] = await once(child, 'close');
      assert.equal(code, 1, errors);
      assert.equal(await readFile(join(root, 'dist', 'src', 'agent.js'), 'utf8'), '// new agent');
      assert.equal(await readFile(join(root, 'node_modules', 'version'), 'utf8'), 'new dependencies');
      assert.equal(JSON.parse(await readFile(join(root, '.releases.json'), 'utf8')).agent, 'new-release');
    }
    assert.equal(uploads, 1);
  } finally {
    if (child) await stopChild(child);
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    await rm(root, { recursive: true, force: true });
  }
});
