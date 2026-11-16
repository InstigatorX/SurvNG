import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { replaceUpdate, validateArchive } from '../src/agent-update.js';

test('a failed multi-component update restores all installed components', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-update-test-'));
  try {
    await writeFile(join(root, 'package.json'), 'old package');
    await mkdir(join(root, 'dist'));
    await writeFile(join(root, 'dist', 'version'), 'old program');
    await writeFile(join(root, 'new-package'), 'new package');
    await assert.rejects(replaceUpdate(root, [
      { name: 'package.json', source: join(root, 'new-package') },
      { name: 'dist', source: join(root, 'missing-stage') },
    ]));
    assert.equal(await readFile(join(root, 'package.json'), 'utf8'), 'old package');
    assert.equal(await readFile(join(root, 'dist', 'version'), 'utf8'), 'old program');
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('update validation rejects links even when every name is allowed', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-archive-test-'));
  try {
    await mkdir(join(root, 'dist'));
    await writeFile(join(root, 'dist', 'regular'), 'program');
    const regular = join(root, 'regular.tgz');
    let child = spawn('tar', ['-czf', regular, '-C', root, 'dist']);
    assert.equal((await once(child, 'close'))[0], 0);
    const allow = (name: string) => name === 'dist/' || name.startsWith('dist/');
    await validateArchive(regular, allow);
    await symlink('/tmp/outside-survng', join(root, 'dist', 'link'));
    const linked = join(root, 'linked.tgz');
    child = spawn('tar', ['-czf', linked, '-C', root, 'dist']);
    assert.equal((await once(child, 'close'))[0], 0);
    await assert.rejects(validateArchive(linked, allow), /rejected/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
