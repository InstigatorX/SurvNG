from __future__ import annotations

import json
import unittest
from io import BytesIO
from urllib.error import HTTPError

from survng.app.ai_provider_transport import (
    AiProviderTransportError,
    request_provider_json,
)


class AiProviderTransportTest(unittest.TestCase):
    @staticmethod
    def _http_error(code: int, payload: dict, headers: dict[str, str] | None = None) -> HTTPError:
        return HTTPError(
            "https://provider.invalid",
            code,
            "provider failure",
            headers or {},
            BytesIO(json.dumps(payload).encode("utf-8")),
        )

    def test_spending_cap_is_actionable_and_is_not_retried(self) -> None:
        calls = 0
        error = self._http_error(429, {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "Your project has exceeded its monthly spending cap. api_key=secret",
            },
        })

        def opener(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise error

        with self.assertRaises(AiProviderTransportError) as raised:
            request_provider_json(
                "https://provider.invalid",
                {},
                {},
                timeout_seconds=5,
                max_response_bytes=1024,
                opener=opener,
                sleeper=lambda _delay: self.fail("spending-cap responses must not sleep"),
            )

        self.assertEqual(calls, 1)
        self.assertEqual(raised.exception.category, "spending_cap")
        self.assertIn("spending cap reached", str(raised.exception))
        self.assertNotIn("secret", str(raised.exception))

    def test_transient_rate_limit_honors_retry_delay_then_succeeds(self) -> None:
        responses = [
            self._http_error(429, {
                "error": {
                    "status": "RESOURCE_EXHAUSTED",
                    "message": "Please retry shortly",
                    "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1.5s"}],
                },
            }),
            BytesIO(b'{"result":"ok"}'),
        ]
        delays: list[float] = []

        def opener(*_args, **_kwargs):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        result = request_provider_json(
            "https://provider.invalid",
            {},
            {},
            timeout_seconds=5,
            max_response_bytes=1024,
            opener=opener,
            sleeper=delays.append,
            jitter=lambda _start, _end: 0.0,
        )

        self.assertEqual(result, {"result": "ok"})
        self.assertEqual(delays, [1.5])

    def test_exhausted_transient_limit_has_bounded_message(self) -> None:
        calls = 0

        def opener(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise self._http_error(429, {
                "error": {"status": "RESOURCE_EXHAUSTED", "message": "temporary capacity"},
            })

        with self.assertRaises(AiProviderTransportError) as raised:
            request_provider_json(
                "https://provider.invalid",
                {},
                {},
                timeout_seconds=5,
                max_response_bytes=1024,
                opener=opener,
                sleeper=lambda _delay: None,
                jitter=lambda _start, _end: 0.0,
            )

        self.assertEqual(calls, 3)
        self.assertEqual(raised.exception.category, "rate_limited")
        self.assertEqual(str(raised.exception), "AI provider is temporarily rate limited; try again shortly")


if __name__ == "__main__":
    unittest.main()
