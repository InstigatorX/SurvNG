import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { lstat, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const service = 'survng-homekit-node.service';
const config = '/etc/survng/homekit-node.env';
const root = '/opt/survng-homekit-node';

async function downloadBundle(work) {
  const mode = await lstat(config);
  if (!mode.isFile() || mode.uid !== 0 || (mode.mode & 0o077)) throw new Error('Encoder credentials are not protected.');
  const lines = (await readFile(config, 'utf8')).split('\n');
  const secret = lines.find(line => line.startsWith('SURVNG_NODE_SECRET='))?.slice('SURVNG_NODE_SECRET='.length);
  const origin = lines.find(line => line.startsWith('SURVNG_ORIGIN='))?.slice('SURVNG_ORIGIN='.length);
  if (!secret || !/^[A-Za-z0-9_-]+$/.test(secret) || !origin) throw new Error('Encoder credentials are incomplete.');
  const url = new URL(origin);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error('Encoder origin is invalid.');
  }
  const base = url.href.replace(/\/$/, '');
  const request = async (path, timeout) => {
    const response = await fetch(base + path, {
      headers: { authorization: 'Bearer ' + secret }, redirect: 'error', signal: AbortSignal.timeout(timeout),
    });
    if (!response.ok || !response.body) throw new Error('Primary rejected the encoder migration download.');
    return response;
  };
  const metadata = await request('/api/homekit/nodes/releases', 10_000);
  const { agent } = await metadata.json();
  if (typeof agent !== 'string' || !/^[0-9a-f]{64}$/.test(agent)) throw new Error('Primary has no managed encoder release.');
  const response = await request('/api/homekit/nodes/agent?release=' + agent, 120_000);
  const archive = join(work, 'agent.tgz');
  const digest = createHash('sha256');
  let length = 0;
  const check = new Transform({ transform(chunk, _encoding, done) {
    length += chunk.length;
    if (length > 512 * 1024 * 1024) return done(new Error('Encoder archive exceeds capacity.'));
    digest.update(chunk);
    done(null, chunk);
  } });
  await pipeline(Readable.fromWeb(response.body), check, createWriteStream(archive, { mode: 0o600 }));
  if (digest.digest('hex') !== agent) throw new Error('Encoder archive digest differs from the pinned release.');
  const options = { timeout: 30_000, maxBuffer: 8 * 1024 * 1024, killSignal: 'SIGKILL' };
  const { stdout: listed } = await run('tar', ['-tzf', archive], options);
  const { stdout: typed } = await run('tar', ['-tvzf', archive, '--numeric-owner'], options);
  const names = listed.split('\n').filter(Boolean);
  const types = typed.split('\n').filter(Boolean);
  if (!names.includes('dist/src/agent-launcher.js') || new Set(names).size !== names.length || types.length !== names.length
      || names.some(name => name.includes('\\') || name.split('/').includes('..')
        || (name !== 'package.json' && !/^(dist|node_modules)\//.test(name)))
      || types.some(line => line[0] !== '-' && line[0] !== 'd')) {
    throw new Error('Encoder archive contents were rejected.');
  }
  await run('tar', ['-xzf', archive, '-C', work, '--no-same-owner', '--no-same-permissions'], options);
  await rm(archive);
  await writeFile(join(work, '.releases.json'), JSON.stringify({ agent }), { mode: 0o600 });
}

async function migrate() {
  if (process.getuid() !== 0 || process.versions.node.split('.')[0] !== '24') {
    throw new Error('Run as root on the encoder using Node.js 24.');
  }
  const work = await mkdtemp(join(tmpdir(), 'survng-encoder-migration-'));
  try {
    await downloadBundle(work);
    await run('systemctl', ['stop', service]);
    try {
      await run(process.execPath, [join(work, 'dist/src/agent-launcher.js'), '--migrate', root, work]);
    } finally {
      await run('systemctl', ['start', service]);
    }
    console.log('Encoder restarted. Confirm it is online and current in Admin → HomeKit.');
  } finally {
    await rm(work, { recursive: true, force: true });
  }
}

async function main() {
  if (process.argv[2] === '--lock-held') {
    await migrate();
    return;
  }
  if (process.getuid() !== 0) throw new Error('Run as root on the encoder using Node.js 24.');
  const { stdout, stderr } = await run('flock', ['-n', '-E', '75', '/run/survng-homekit-encoder-migration.lock',
    process.execPath, fileURLToPath(import.meta.url), '--lock-held']);
  process.stdout.write(stdout);
  process.stderr.write(stderr);
}

main().catch(error => {
  if (error.code === 75) {
    console.error('Another encoder migration is running.');
  } else if (error.stderr && process.argv[2] !== '--lock-held') {
    process.stderr.write(error.stderr);
  } else {
    console.error(error.cmd
      ? 'Encoder migration command failed. Check the primary release, disk space and encoder service journal.'
      : error.message);
  }
  process.exitCode = 1;
});
