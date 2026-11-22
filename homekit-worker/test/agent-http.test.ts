import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { networkInterfaces } from 'node:os';
import { PassThrough, Readable } from 'node:stream';
import { mediaUploadOrigin, post } from '../src/agent-http.js';

test('media routing accepts a private LAN listener or an HTTPS override', () => {
  const enrolled = 'https://survng.example/cameras';
  assert.equal(mediaUploadOrigin(enrolled, 'https://[::1]:8443/new'), 'https://[::1]:8443/new');
  for (const address of ['10.20.30.40', '172.16.0.1', '172.31.0.1', '192.168.86.243']) {
    const lan = `http://${address}:8088/cameras`;
    assert.equal(mediaUploadOrigin(enrolled, lan), lan);
  }
  assert.equal(mediaUploadOrigin(enrolled, null), enrolled);
  assert.equal(mediaUploadOrigin(enrolled, undefined), enrolled);
  assert.equal(mediaUploadOrigin('http://survng.example/cameras', null), 'http://survng.example/cameras');
});

test('media routing rejects public and ambiguous HTTP destinations and unsafe URL components', () => {
  const enrolled = 'https://survng.example/cameras';
  for (const invalid of ['http://survng.example/cameras', 'http://203.0.113.1', 'http://127.0.0.1', 'http://172.32.0.1',
    'http://0xc0.0xa8.1.1', 'http://192.168.1.1:0', 'https://user:pw@media.example',
    'https://media.example?redirect=somewhere', 'https://media.example/#', '//media.example',
    'https:///media.example', 'https://media.example/a/../b', 'https://media.example/a//b', 'https://media.example/%2f', 123]) {
    assert.throws(() => mediaUploadOrigin(enrolled, invalid));
  }
  assert.throws(() => mediaUploadOrigin('http://survng.example', 'http://media.example'));
});

test('the LAN endpoint receives video before the upload ends', { timeout: 5000 }, async t => {
  const address = Object.values(networkInterfaces()).flat().find(info =>
    info?.family === 'IPv4' && /^(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)/.test(info.address))?.address;
  if (!address) return t.skip('requires a private IPv4 interface');

  const abort = new AbortController();
  const body = new PassThrough();
  let receiveVideo!: (data: string) => void;
  const video = new Promise<string>(resolve => { receiveVideo = resolve; });
  const lan = createServer(request => {
    request.once('data', chunk => receiveVideo(chunk.toString()));
  });
  const controlPaths: string[] = [];
  const control = createServer((request, response) => {
    controlPaths.push(request.url!);
    const listener = lan.address();
    assert.ok(listener && typeof listener !== 'string');
    response.end(JSON.stringify({ ingest: `http://${address}:${listener.port}/survng` }));
  });
  t.after(() => {
    abort.abort();
    body.destroy();
    for (const server of [control, lan]) { server.closeAllConnections(); server.close(); }
  });

  lan.listen(0, address);
  control.listen(0, '127.0.0.1');
  await Promise.all([once(lan, 'listening'), once(control, 'listening')]);
  const listener = control.address();
  assert.ok(listener && typeof listener !== 'string');
  const enrolled = `http://127.0.0.1:${listener.port}/survng`;
  const sync = await post(enrolled + '/api/homekit/nodes/sync', '{}', {}, abort.signal);
  const origin = mediaUploadOrigin(enrolled, (await sync.json()).ingest);
  const upload = post(origin + '/api/homekit/nodes/media', body, {}, abort.signal, true).catch(() => {});
  body.write('video fragment');
  assert.equal(await video, 'video fragment');
  assert.equal(body.writableEnded, false);
  assert.deepEqual(controlPaths, ['/survng/api/homekit/nodes/sync']);
  abort.abort();
  await upload;
});

// Measure live buffers in a fresh process so other media fixtures and allocator
// high-water marks cannot hide retention. The upload deliberately never ends.
test('a continuous upload does not retain all previously sent camera bytes', { timeout: 15000 }, async () => {
  const moduleUrl = new URL('../src/agent-http.js', import.meta.url).href;
  const child = spawn(process.execPath, ['--expose-gc', '--input-type=module', '-e', `
    import { createServer } from 'node:http';
    import { networkInterfaces } from 'node:os';
import { PassThrough, Readable } from 'node:stream';
    import { once } from 'node:events';
    import { post } from ${JSON.stringify(moduleUrl)};
    let sent = 0, received = 0, measured = false;
    const total = 64 * 1024 * 1024;
    const abort = new AbortController();
    const source = new Readable({ read() {
      if (sent < total) { sent += 65536; this.push(Buffer.alloc(65536, 0x55)); }
    } });
    source.on('error', () => {});
    const server = createServer(request => request.on('data', chunk => {
      received += chunk.length;
      if (received === total && !measured) {
        measured = true;
        setTimeout(() => {
          global.gc(); global.gc();
          console.log(JSON.stringify({ received, ended: source.readableEnded, ...process.memoryUsage() }));
          abort.abort(); source.destroy(); server.closeAllConnections(); server.close();
        }, 30);
      }
    }));
    server.listen(0, '127.0.0.1'); await once(server, 'listening');
    const deadline = setTimeout(() => {
      process.exitCode = 1; abort.abort(); source.destroy(); server.closeAllConnections(); server.close();
    }, 10000);
    await post('http://127.0.0.1:' + server.address().port, source, {}, abort.signal, true).catch(() => {});
    clearTimeout(deadline);
  `], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '', stderr = '';
  child.stdout.on('data', chunk => stdout += chunk.toString());
  child.stderr.on('data', chunk => stderr += chunk.toString());
  const [code] = await once(child, 'close');
  assert.equal(code, 0, stderr);
  const sample = JSON.parse(stdout);
  assert.equal(sample.received, 64 * 1024 * 1024);
  assert.equal(sample.ended, false, 'measure while the camera upload is still open');
  assert.ok(sample.arrayBuffers < 8 * 1024 * 1024,
    `retained ${(sample.arrayBuffers / 1024 ** 2).toFixed(1)} MiB after sending 64 MiB`);
});

test('a media upload rejects a redirect without replaying it elsewhere', async () => {
  const requests: string[] = [];
  const server = createServer((request, response) => {
    requests.push(request.url!); request.resume();
    response.writeHead(303, { location: '/elsewhere' }); response.end();
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const port = (server.address() as { port: number }).port;
  const body = Readable.from([Buffer.from('synthetic media')]);
  try {
    await assert.rejects(post(`http://127.0.0.1:${port}/upload`, body, {}, AbortSignal.timeout(2000), true));
    assert.deepEqual(requests, ['/upload']);
  } finally {
    body.destroy(); server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
  }
});
