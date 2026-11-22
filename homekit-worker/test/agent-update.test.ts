import { createHash } from 'node:crypto';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, symlink } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { validateArchive, verifyToolRelease } from '../src/agent-update.js';

test('the FFmpeg release covers both staged executables', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-tools-test-'));
  try {
    await writeFile(join(root, 'ffmpeg'), 'encoder');
    await writeFile(join(root, 'ffprobe'), 'probe');
    const digest = createHash('sha256').update('encoderprobe').digest('hex');
    await verifyToolRelease('ffmpeg', digest, root, new AbortController().signal);
    await writeFile(join(root, 'ffprobe'), 'stale probe');
    await assert.rejects(verifyToolRelease('ffmpeg', digest, root, new AbortController().signal), /release mismatch/);
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
