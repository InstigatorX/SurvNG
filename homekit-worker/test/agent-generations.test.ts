import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import { copyFile, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { test, type TestContext } from 'node:test';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';
import { activateGeneration, discardGeneration, migrateGeneration, stageGeneration } from '../src/agent-generations.js';
import { generationStore, launchEncoder, readBoot } from '../src/agent-launcher.js';

const oldDigest = '1'.repeat(64);
const newDigest = '2'.repeat(64);
const oldSource = `
  import fs from 'node:fs';
  fs.writeFileSync(process.env.SURVNG_ENCODER_HOME + '/ran', JSON.stringify({
    cwd: process.cwd(), rejected: JSON.parse(process.env.SURVNG_ENCODER_REJECTED), secret: process.env.SURVNG_NODE_SECRET
  }));
`;

async function scaffold(root: string, code: string, digest: string): Promise<void> {
  await mkdir(join(root, 'dist/src'), { recursive: true });
  await mkdir(join(root, 'node_modules'));
  await writeFile(join(root, 'package.json'), '{"type":"module"}');
  await writeFile(join(root, 'dist/src/agent.js'), code, { mode: 0o750 });
  for (const name of ['agent-launcher', 'agent-generations']) {
    await copyFile(fileURLToPath(new URL('../src/' + name + '.js', import.meta.url)), join(root, 'dist/src/' + name + '.js'));
  }
  await writeFile(join(root, '.releases.json'), JSON.stringify({ agent: digest, ffmpeg: 'tool1', runtime: 'lib1' }));
}

async function installation(t: TestContext, startup: string, before = '') {
  const temp = await mkdtemp(join(tmpdir(), 'survng-encoder-test-'));
  t.after(() => rm(temp, { recursive: true, force: true }));
  const root = join(temp, 'install');
  const bundle = join(temp, 'bundle');
  await scaffold(root, oldSource, oldDigest);
  for (const dir of ['bin', 'lib']) await mkdir(join(root, dir));
  await writeFile(join(root, 'bin/ffmpeg'), 'current tools');
  await writeFile(join(root, 'lib/library'), 'current libraries');
  await scaffold(bundle, `
    if (process.argv[2] === '--preflight') {
      if (process.env.SURVNG_NODE_SECRET) process.exit(32);
      ${before}
      process.exit(0);
    }
    ${startup}
  `, newDigest);
  return { root, bundle };
}

async function run(script: string, root: string, args: string[] = []): Promise<number | null> {
  const child = spawn(process.execPath, [script, ...args], { cwd: root,
    env: { ...process.env, SURVNG_NODE_SECRET: 'fixture-secret' }, stdio: 'ignore' });
  return (await once(child, 'close'))[0] as number | null;
}
const fixedEntry = (root: string) => join(root, 'dist/src/agent.js');

test('local migration preserves legacy entry, tools and service credentials through a failed trial', async t => {
  const { root, bundle } = await installation(t, 'process.exit(13)');
  const launcher = join(bundle, 'dist/src/agent-launcher.js');
  assert.equal(await run(launcher, root, ['--migrate', root, bundle]), 0);
  const queued = readBoot(root);
  assert.equal(queued.trial?.attempted, false);
  const good = join(root, generationStore, queued.good);
  const candidate = join(root, generationStore, queued.trial!.id);
  assert.equal(await readFile(fixedEntry(good), 'utf8'), oldSource);
  assert.equal((await stat(fixedEntry(root))).mode & 0o777, 0o750);
  assert.equal(await readFile(join(candidate, 'bin/ffmpeg'), 'utf8'), 'current tools');
  assert.deepEqual(JSON.parse(await readFile(join(candidate, '.releases.json'), 'utf8')),
    { agent: newDigest, ffmpeg: 'tool1', runtime: 'lib1' });
  await migrateGeneration(root, bundle);
  assert.deepEqual(readBoot(root), queued);
  assert.equal(await run(fixedEntry(root), root), 1);
  assert.equal(readBoot(root).good, queued.good);
  assert.equal(await run(fixedEntry(root), root), 0);
  const ran = JSON.parse(await readFile(join(root, 'ran'), 'utf8'));
  assert.equal(ran.cwd, good);
  assert.equal(ran.secret, 'fixture-secret');
  assert.equal(ran.rejected.agent, newDigest);
  await assert.rejects(stat(candidate), { code: 'ENOENT' });
});

test('preflight failure leaves fixed legacy boot intact and no trial published', async t => {
  const { root, bundle } = await installation(t, '', 'process.exit(3);');
  await assert.rejects(migrateGeneration(root, bundle), /preflight/);
  assert.equal(await readFile(fixedEntry(root), 'utf8'), oldSource);
  assert.equal(readBoot(root).trial, undefined);
});

test('a migration retry clears unselected files left before the first boot record', async t => {
  const { root, bundle } = await installation(t, "process.send('ready');");
  const abandoned = join(root, generationStore, 'gen-orphan');
  const scratch = join(root, '.encoder-migrate-orphan');
  await mkdir(abandoned, { recursive: true });
  await mkdir(scratch);
  await migrateGeneration(root, bundle);
  assert.ok(readBoot(root).trial);
  await assert.rejects(stat(abandoned), { code: 'ENOENT' });
  await assert.rejects(stat(scratch), { code: 'ENOENT' });
});

test('candidate preflight cannot load a package from the fixed legacy installation', async t => {
  const { root, bundle } = await installation(t, '', "await import('legacy-only');");
  const dependency = join(root, 'node_modules/legacy-only');
  await mkdir(dependency);
  await writeFile(join(dependency, 'package.json'), '{"main":"index.js"}');
  await writeFile(join(dependency, 'index.js'), 'module.exports = {};');
  await assert.rejects(migrateGeneration(root, bundle), /preflight/);
  assert.equal(await readFile(fixedEntry(root), 'utf8'), oldSource);
  assert.equal(readBoot(root).trial, undefined);
});

test('launcher recovers a hard interruption after recording the attempt and before readiness', async t => {
  const { root, bundle } = await installation(t, `
    const fs = await import('node:fs');
    process.on('disconnect', () => process.exit(0));
    fs.writeFileSync(process.env.SURVNG_ENCODER_HOME + '/started', String(process.pid));
    setInterval(() => {}, 1000);
  `);
  await migrateGeneration(root, bundle);
  const queued = readBoot(root);
  const launcher = spawn(process.execPath, [fixedEntry(root)], { cwd: root, stdio: 'ignore' });
  const closed = once(launcher, 'close');
  try {
    let started = false;
    for (let i = 0; i < 200 && !started; i++) {
      try { await stat(join(root, 'started')); started = true; }
      catch { await delay(10); }
    }
    assert.ok(started, 'synthetic trial reached startup');
    assert.equal(readBoot(root).trial?.attempted, true);
  } finally { launcher.kill('SIGKILL'); await closed; }
  assert.equal(await run(fixedEntry(root), root), 0);
  assert.deepEqual(readBoot(root), {
    version: 1, good: queued.good, rejected: { agent: newDigest, ffmpeg: 'tool1', runtime: 'lib1' },
  });
  assert.equal(JSON.parse(await readFile(join(root, 'ran'), 'utf8')).cwd, join(root, generationStore, queued.good));
});

test('a stalled trial is terminated and its unchanged tools, stamp and request are recovered coherently', async t => {
  const { root } = await installation(t, '');
  await writeFile(fixedEntry(root), "if (process.argv[2] !== '--preflight') process.send?.('ready');");
  await migrateGeneration(root, root);
  await migrateGeneration(root, root);
  const good = join(root, generationStore, readBoot(root).good);
  const source = join(root, 'new-dist');
  await mkdir(join(source, 'src'), { recursive: true });
  await writeFile(join(source, 'src/agent.js'), 'setInterval(() => {}, 1000);');
  const stamp = join(root, 'candidate-stamp');
  const releases = { agent: newDigest, ffmpeg: 'tool1', runtime: 'lib1', syncRequest: '3'.repeat(32) };
  await writeFile(stamp, JSON.stringify(releases));
  const candidate = await stageGeneration(root, good, [{ name: 'dist', source }, { name: '.releases.json', source: stamp }]);
  await activateGeneration(root, candidate);
  await discardGeneration(root, candidate);
  assert.equal(await launchEncoder(root, 100), 1);
  assert.deepEqual(readBoot(root), { version: 1, good: basename(good), rejected: releases });
  assert.equal(await readFile(join(good, 'lib/library'), 'utf8'), 'current libraries');
  await discardGeneration(root, candidate);
  await assert.rejects(stat(candidate), { code: 'ENOENT' });
});

test('only a ready trial is promoted, subsequent service boots stay on that generation', async t => {
  const { root, bundle } = await installation(t, "process.send('ready');");
  await migrateGeneration(root, bundle);
  const selected = readBoot(root).trial!.id;
  assert.equal(await run(fixedEntry(root), root), 0);
  assert.deepEqual(readBoot(root), { version: 1, good: selected });
  assert.equal(await run(fixedEntry(root), root), 0);
});

test('a trial stays selected if directory sync reports failure after rename', async t => {
  const { root } = await installation(t, '');
  await writeFile(fixedEntry(root), "if (process.argv[2] !== '--preflight') process.send?.('ready');");
  await migrateGeneration(root, root);
  const good = join(root, generationStore, readBoot(root).good);
  const candidate = await stageGeneration(root, good, []);
  const realSync = fs.fsyncSync;
  const injected = t.mock.method(fs, 'fsyncSync', (descriptor: number) => {
    if (fs.fstatSync(descriptor).isDirectory()) throw new Error('synthetic directory sync failure');
    return realSync(descriptor);
  });
  syncBuiltinESMExports();
  try {
    await activateGeneration(root, candidate);
    assert.equal(readBoot(root).trial?.id, basename(candidate));
  } finally {
    injected.mock.restore();
    syncBuiltinESMExports();
  }
});

test('cold startup removes only interrupted updater scratch directories', async t => {
  const { root } = await installation(t, '');
  await writeFile(fixedEntry(root), "if (process.argv[2] !== '--preflight') process.send?.('ready');");
  await migrateGeneration(root, root);
  const archives = await mkdtemp(join(root, '.update-'));
  const stamp = await mkdtemp(join(root, '.update-stamp-'));
  const keep = join(root, '.update-operator-backup');
  await mkdir(keep);
  assert.equal(await run(fixedEntry(root), root), 0);
  for (const path of [archives, stamp]) await assert.rejects(stat(path), { code: 'ENOENT' });
  assert.ok((await stat(keep)).isDirectory());
});
