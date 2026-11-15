import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { Readable } from 'node:stream';
import { post } from '../src/agent-http.js';

// Measure live buffers in a fresh process so other media fixtures and allocator
// high-water marks cannot hide retention. The upload deliberately never ends.
test('a continuous upload does not retain all previously sent camera bytes', { timeout: 15000 }, async () => {
  const moduleUrl = new URL('../src/agent-http.js', import.meta.url).href;
  const child = spawn(process.execPath, ['--expose-gc', '--input-type=module', '-e', `
    import { createServer } from 'node:http';
    import { Readable } from 'node:stream';
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
