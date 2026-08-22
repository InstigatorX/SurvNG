from __future__ import annotations

import asyncio
import gzip
import unittest

from survng.app.main import JsonGZipMiddleware


class JsonResponseCompressionTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _receive() -> dict:
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    @staticmethod
    def _collector(messages: list[dict]):
        async def collect(message: dict) -> None:
            messages.append(message)

        return collect

    async def _request(
        self,
        *,
        content_type: bytes,
        body: bytes,
        accept_encoding: bytes = b"gzip",
    ) -> list[dict]:
        async def inner(_scope, _receive, send) -> None:
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", content_type), (b"content-length", str(len(body)).encode())],
            })
            await send({"type": "http.response.body", "body": body})

        messages: list[dict] = []
        middleware = JsonGZipMiddleware(inner, minimum_size=64, compresslevel=5)
        await middleware({
            "type": "http", "method": "GET", "path": "/api/example",
            "headers": [(b"accept-encoding", accept_encoding)],
        }, self._receive, self._collector(messages))
        return messages

    async def test_large_json_is_gzipped_and_varies_by_accept_encoding(self) -> None:
        body = b'{"items":["' + (b"camera-event," * 40) + b'"]}'

        messages = await self._request(content_type=b"application/json", body=body)

        headers = dict(messages[0]["headers"])
        self.assertEqual(headers[b"content-encoding"], b"gzip")
        self.assertIn(b"accept-encoding", headers[b"vary"].lower())
        self.assertEqual(gzip.decompress(messages[1]["body"]), body)

    async def test_sse_and_media_bodies_are_not_gzipped(self) -> None:
        body = b"x" * 512
        for content_type in (b"text/event-stream", b"video/mp4", b"multipart/x-mixed-replace"):
            with self.subTest(content_type=content_type):
                messages = await self._request(content_type=content_type, body=body)
                headers = dict(messages[0]["headers"])
                self.assertNotIn(b"content-encoding", headers)
                self.assertNotIn(b"vary", headers)
                self.assertEqual(messages[1]["body"], body)

    async def test_small_json_remains_uncompressed(self) -> None:
        messages = await self._request(content_type=b"application/json", body=b'{"ok":true}')

        headers = dict(messages[0]["headers"])
        self.assertNotIn(b"content-encoding", headers)
        self.assertNotIn(b"vary", headers)

    async def test_gzip_quality_zero_and_substring_codings_are_not_accepted(self) -> None:
        body = b'{"items":["' + (b"camera-event," * 40) + b'"]}'
        for accept_encoding in (
            b"gzip;q=0",
            b"br, gzip; q=0.000",
            b"x-gzipish",
            b"*;q=1, gzip;q=0",
            b"gzip;q=invalid",
        ):
            with self.subTest(accept_encoding=accept_encoding):
                messages = await self._request(
                    content_type=b"application/json",
                    body=body,
                    accept_encoding=accept_encoding,
                )
                headers = dict(messages[0]["headers"])
                self.assertNotIn(b"content-encoding", headers)
                self.assertEqual(messages[1]["body"], body)

    async def test_positive_gzip_quality_and_wildcard_are_accepted(self) -> None:
        body = b'{"items":["' + (b"camera-event," * 40) + b'"]}'
        for accept_encoding in (b"br, GZip; q=0.25", b"br, *;q=0.5"):
            with self.subTest(accept_encoding=accept_encoding):
                messages = await self._request(
                    content_type=b"application/json",
                    body=body,
                    accept_encoding=accept_encoding,
                )
                headers = dict(messages[0]["headers"])
                self.assertEqual(headers[b"content-encoding"], b"gzip")
                self.assertEqual(gzip.decompress(messages[1]["body"]), body)

    async def test_streaming_json_chunks_are_awaited_and_gzipped(self) -> None:
        """Starlette 1.6+ apply_compression is async; chunked bodies must await it."""
        chunks = [b'{"items":[', b'"' + (b"x" * 80) + b'",', b'"' + (b"y" * 80) + b'"]}']

        async def inner(_scope, _receive, send) -> None:
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(sum(len(chunk) for chunk in chunks)).encode()),
                ],
            })
            for index, chunk in enumerate(chunks):
                await send({
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": index < len(chunks) - 1,
                })

        messages: list[dict] = []
        middleware = JsonGZipMiddleware(inner, minimum_size=64, compresslevel=5)
        await middleware(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/example",
                "headers": [(b"accept-encoding", b"gzip")],
            },
            self._receive,
            self._collector(messages),
        )

        headers = dict(messages[0]["headers"])
        self.assertEqual(headers[b"content-encoding"], b"gzip")
        self.assertNotIn(b"content-length", headers)
        compressed = b"".join(message["body"] for message in messages[1:])
        self.assertEqual(gzip.decompress(compressed), b"".join(chunks))


if __name__ == "__main__":
    unittest.main()
