import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer, type ServerResponse } from 'node:http';
import { execFile, spawn, type ChildProcess } from 'node:child_process';
import { once } from 'node:events';
import { appendFile, cp, lstat, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { generationStore, readBoot } from '../src/agent-launcher.js';
import { stopChild } from '../src/media.js';

const run = promisify(execFile);
const oldRelease = '0'.repeat(64);
const firstNonce = '1'.repeat(32);
const secondNonce = '2'.repeat(32);
const secret = 'test-only-agent-secret';
const agentEntry = 'dist/src/agent.js';
const ffprobe = `#!/bin/sh
case "$1" in
  -version) echo 'ffprobe version fixture' ;;
  *) echo '{"streams":[{"width":1280,"height":720}]}' ;;
esac
`;
function ffmpeg(version: string, withOpus = true): string {
  return `#!/bin/sh
case " $* " in
  *' -version '*) echo 'ffmpeg version ${version}' ;;
  *' -encoders '*) printf '%s\\n' ' V..... libx265' ' A..... aac' ' V..... mjpeg' ${withOpus ? "' A..... libopus'" : ''} ;;
  *' image2pipe '*) printf 'fixture snapshot' ;;
  *) exit 1 ;;
esac
`;
}

async function archive(directory: string, members: string[]): Promise<Buffer> {
  const file = join(directory, 'payload.tgz');
  await run('tar', ['-czf', file, '-C', directory, ...members]);
  return readFile(file);
}

type Report = { runningVersion: string; failedSyncRequest?: string; ffmpeg: string };
for (const scenario of ['held download', 'missing ffprobe', 'missing encoder', 'wrong digest', 'manual retry'] as const) {
  test(`encoder service update: ${scenario}`, { timeout: 30000 }, async () => {
    const fixture = await mkdtemp(join(tmpdir(), 'survng-agent-process-'));
    const root = join(fixture, 'install');
    const stage = join(fixture, 'agent');
    const tools = join(fixture, 'tools');
    const server = createServer();
    let child: ChildProcess | undefined;
    let errors = '';
    let held: ServerResponse | undefined;
    try {
      // Match the published package, which includes regular files and directories.
      await cp(fileURLToPath(new URL('../src', import.meta.url)), join(stage, 'dist/src'), { recursive: true });
      await cp(fileURLToPath(new URL('../../node_modules', import.meta.url)), join(stage, 'node_modules'), {
        recursive: true, filter: async source => !(await lstat(source)).isSymbolicLink(),
      });
      await writeFile(join(stage, 'package.json'), '{"type":"module"}');
      await cp(stage, root, { recursive: true });
      await mkdir(join(root, 'bin'));
      await mkdir(join(root, 'lib'));
      await writeFile(join(root, 'bin/ffmpeg'), ffmpeg('initial'), { mode: 0o755 });
      await writeFile(join(root, 'bin/ffprobe'), ffprobe, { mode: 0o755 });
      await writeFile(join(root, '.releases.json'), JSON.stringify({ agent: oldRelease }));
      const initialProgram = await readFile(join(root, agentEntry), 'utf8');
      await run(process.execPath, [join(root, 'dist/src/agent-launcher.js'), '--migrate', root, root]);
      const initialBoot = readBoot(root);
      const original = join(root, generationStore, initialBoot.good);
      await appendFile(join(stage, agentEntry), '\n// candidate build\n');
      const agent = await archive(stage, ['dist', 'package.json', 'node_modules']);
      const release = scenario === 'wrong digest' ? 'f'.repeat(64) : createHash('sha256').update(agent).digest('hex');
      await mkdir(tools);
      const toolProgram = ffmpeg('candidate', scenario !== 'missing encoder');
      await writeFile(join(tools, 'ffmpeg'), toolProgram, { mode: 0o755 });
      await writeFile(join(tools, 'ffprobe'), ffprobe, { mode: 0o755 });
      const toolArchive = await archive(tools, scenario === 'missing ffprobe' ? ['ffmpeg'] : ['ffmpeg', 'ffprobe']);
      const toolRelease = createHash('sha256').update(toolProgram).update(ffprobe).digest('hex');
      const reports: Report[] = [];
      const snapshots: string[] = [];
      let downloads = 0;
      let toolDownloads = 0;
      let nonce = scenario === 'held download' ? undefined : firstNonce;
      let serverFailure: unknown;
      server.on('request', async (req, res) => {
        try {
          assert.equal(req.headers.authorization, `Bearer ${secret}`);
          if (req.url === '/api/homekit/nodes/sync') {
            const chunks: Buffer[] = [];
            for await (const chunk of req) chunks.push(chunk);
            reports.push(JSON.parse(Buffer.concat(chunks).toString('utf8')));
            // Issue this assignment only after the server has received and held the archive request.
            const assignments = held ? [{ id: 'front', url: 'fixture-camera', transport: 'tcp', revision: 'r1',
              audio: false, allowSoftwareEncoding: false }] : [];
            await delay(20);
            res.setHeader('content-type', 'application/json');
            res.end(JSON.stringify({ assignments, feeds: [], snapshots: held ? ['front'] : [],
              snapshotRequests: held ? { front: 'during-update' } : {}, syncRequest: nonce,
              releases: { agent: release, ffmpeg: toolRelease } }));
          } else if (req.url === `/api/homekit/nodes/agent?release=${release}`) {
            downloads++;
            if (scenario === 'held download') {
              held = res;
              res.write(agent.subarray(0, 32));
            } else if (scenario === 'manual retry' && downloads === 1) {
              res.writeHead(503).end();
            } else res.end(agent);
          } else if (req.url === '/api/homekit/nodes/ffmpeg') {
            toolDownloads++;
            res.end(toolArchive);
          } else if (req.url === '/api/homekit/nodes/snapshot') {
            assert.equal(req.headers['x-survng-camera'], 'front');
            assert.equal(req.headers['x-survng-request'], 'during-update');
            const chunks: Buffer[] = [];
            for await (const chunk of req) chunks.push(chunk);
            snapshots.push(Buffer.concat(chunks).toString('utf8'));
            res.end();
          } else throw new Error(`unexpected request ${req.url}`);
        } catch (error) { serverFailure = error; res.destroy(); }
      });
      server.listen(0, '127.0.0.1');
      await once(server, 'listening');
      const port = (server.address() as { port: number }).port;
      function launch(): ChildProcess {
        const service = spawn(process.execPath, [agentEntry], { cwd: root, env: { ...process.env,
          SURVNG_ORIGIN: `http://127.0.0.1:${port}`, SURVNG_NODE_SECRET: secret, SURVNG_ENCODER_HOME: root,
          DEBUG: '', NODE_DEBUG: '' }, stdio: ['ignore', 'ignore', 'pipe'] });
        service.stderr.on('data', chunk => { errors = (errors + chunk).slice(-4096); });
        return service;
      }
      async function waitUntil(predicate: () => boolean, outcome: string): Promise<void> {
        const deadline = Date.now() + 10000;
        while (!predicate()) {
          if (serverFailure) throw serverFailure;
          if (child?.exitCode !== null || child.signalCode !== null || Date.now() > deadline) {
            assert.fail(`waiting for ${outcome}, exit=${child?.exitCode}, stderr=${errors}`);
          }
          await delay(20);
        }
      }
      child = launch();
      if (scenario === 'held download') {
        await waitUntil(() => snapshots.length > 0, 'snapshot assignment while download is held');
        assert.deepEqual(snapshots, ['fixture snapshot']);
        assert.deepEqual(readBoot(root), initialBoot);
        held!.end(agent.subarray(32));
        held = undefined;
      } else {
        // Persisting the same nonce across heartbeats must report the failure without retrying it.
        await waitUntil(() => reports.filter(report => report.failedSyncRequest === firstNonce).length >= 2, 'failed nonce on two heartbeats');
        assert.equal(downloads, 1);
        assert.deepEqual(readBoot(root), initialBoot);
        if (scenario === 'manual retry') nonce = secondNonce;
      }
      if (scenario === 'held download' || scenario === 'manual retry') {
        await waitUntil(() => child?.exitCode !== null, 'exit requesting a service restart');
        assert.equal(child.exitCode, 1, errors);
        const { good, trial } = readBoot(root);
        assert.equal(good, initialBoot.good);
        assert.ok(trial && !trial.attempted);
        const candidate = join(root, generationStore, trial.id);
        assert.equal(await readFile(join(candidate, agentEntry), 'utf8'), initialProgram + '\n// candidate build\n');
        assert.equal(await readFile(join(candidate, 'bin/ffmpeg'), 'utf8'), toolProgram);
        assert.ok(reports.length && reports.every(report => report.runningVersion === oldRelease));
        child = launch(); // The unit restarts the stable entry after the old process exits.
        await waitUntil(() => reports.some(report => report.runningVersion === release && report.ffmpeg === 'ffmpeg version candidate')
          && readBoot(root).good === trial.id, 'confirmed candidate and its first heartbeat');
        assert.equal(readBoot(root).trial, undefined);
      } else {
        assert.ok(reports.every(report => report.runningVersion === oldRelease && report.ffmpeg === 'ffmpeg version initial'));
      }
      assert.equal(await readFile(join(original, agentEntry), 'utf8'), initialProgram);
      assert.equal(await readFile(join(original, 'bin/ffmpeg'), 'utf8'), ffmpeg('initial'));
      assert.deepEqual(JSON.parse(await readFile(join(original, '.releases.json'), 'utf8')), { agent: oldRelease });
      assert.equal(downloads, scenario === 'manual retry' ? 2 : 1);
      assert.equal(toolDownloads, scenario === 'wrong digest' ? 0 : 1);
      if (serverFailure) throw serverFailure;
    } finally {
      held?.destroy();
      if (child) await stopChild(child);
      server.closeAllConnections();
      await new Promise<void>(resolve => server.close(() => resolve()));
      await rm(fixture, { recursive: true, force: true });
    }
  });
}
