from __future__ import annotations

import socket
import struct
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from survng.app.baichuan_native import (
    BaichuanError,
    BaichuanFfmpegPipe,
    BaichuanNativeClient,
    IFRAME_FIRST,
    MediaFrameReader,
)
from survng.app.config import CameraConfig
from survng.app.onvif_events import OnvifEventListener


def camera(*, onvif: bool = False) -> CameraConfig:
    return CameraConfig.model_validate(
        {
            "id": "gate",
            "name": "Gate",
            "stream_url": "rtsp://example.invalid/main",
            "onvif": {
                "enabled": onvif,
                "host": "example.invalid",
                "username": "admin",
            },
            "baichuan": {
                "enabled": True,
                "host": "example.invalid",
                "username": "admin",
            },
        }
    )


class CaptureProtocolTest(unittest.TestCase):
    def test_native_client_close_interrupts_and_releases_socket(self) -> None:
        client = BaichuanNativeClient(camera())
        sock = Mock()
        client._sock = sock

        client.close()
        client.close()

        sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        sock.close.assert_called_once_with()
        self.assertIsNone(client._sock)

    def test_native_client_does_not_connect_when_already_stopped(self) -> None:
        client = BaichuanNativeClient(camera())
        stop = threading.Event()
        stop.set()

        with patch("survng.app.baichuan_native.socket.create_connection") as connect:
            self.assertEqual(list(client.video_frames(stop)), [])

        connect.assert_not_called()

    def test_baichuan_pipe_stop_closes_client_and_stdin(self) -> None:
        entered = threading.Event()
        fake_client = Mock()

        def video_bytes(stop):
            entered.set()
            while not stop.wait(0.01):
                pass
            if False:
                yield b""

        fake_client.video_bytes.side_effect = video_bytes
        stdin = Mock()
        pipe = BaichuanFfmpegPipe(camera(), "live")
        with patch("survng.app.baichuan_native.BaichuanNativeClient", return_value=fake_client):
            pipe.start(stdin)
            self.assertTrue(entered.wait(1))
            pipe.stop()

        fake_client.close.assert_called()
        stdin.close.assert_called()
        self.assertIsNone(pipe.thread)

    def test_media_reader_rejects_unbounded_extension_before_reading_it(self) -> None:
        header = b"H264" + struct.pack("<III", 1, 2 * 1024 * 1024, 0)
        reader = MediaFrameReader(iter([struct.pack("<I", IFRAME_FIRST) + header]))

        with self.assertRaisesRegex(BaichuanError, "extension length"):
            reader.read_frame()

    def test_onvif_callback_failure_does_not_force_resubscription(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        listener._stop.clear()
        notification = SimpleNamespace(Topic="motion", Message="motion")
        response = SimpleNamespace(NotificationMessage=[notification])
        pullpoint = Mock()

        def pull_messages(_request):
            listener._stop.set()
            return response

        pullpoint.PullMessages.side_effect = pull_messages
        callback = Mock(side_effect=RuntimeError("application failure"))
        listener.on_motion = callback
        fake_onvif = type("FakeOnvifCamera", (), {})
        modules = {
            "onvif": SimpleNamespace(ONVIFCamera=fake_onvif),
            "zeep": SimpleNamespace(Transport=object),
            "zeep.cache": SimpleNamespace(SqliteCache=object),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(listener, "_subscribe", return_value=pullpoint) as subscribe,
            patch.object(listener, "_unsubscribe"),
            patch.object(listener, "_close_transport"),
        ):
            listener._run_until_stopped()

        callback.assert_called_once()
        subscribe.assert_called_once()
        self.assertEqual(listener.poll_errors, 0)
        self.assertEqual(listener.notifications_received, 1)
        self.assertEqual(listener.motion_events_received, 1)
        self.assertEqual(listener.callback_errors, 1)
        self.assertTrue(listener.last_event_at)
        self.assertTrue(listener.last_motion_event_at)

    def test_onvif_run_always_releases_subscription_and_transport(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        manager = Mock()
        transport = SimpleNamespace(session=Mock())
        listener._subscription_manager = manager
        listener._transport = transport

        with patch.object(listener, "_run_until_stopped", return_value=None):
            listener._run()

        manager.Unsubscribe.assert_called_once_with()
        transport.session.close.assert_called_once_with()
        self.assertFalse(listener.connected)

    def test_onvif_clean_stop_unsubscribes_before_closing_transport(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        manager = Mock()
        transport = SimpleNamespace(session=Mock())
        listener._subscription_manager = manager
        listener._transport = transport
        order: list[str] = []
        manager.Unsubscribe.side_effect = lambda: order.append("unsubscribe")
        transport.session.close.side_effect = lambda: order.append("close")

        with patch.object(
            listener,
            "_run_until_stopped",
            side_effect=lambda: listener._stop.wait(2),
        ):
            listener.start()
            listener.stop()

        self.assertEqual(order, ["unsubscribe", "close"])
        self.assertFalse(listener.running)

    def test_onvif_numeric_off_state_is_not_reported_as_motion(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())

        self.assertFalse(
            listener._is_motion_event(
                "tns1:RuleEngine/CellMotionDetector/Motion",
                '<tt:SimpleItem Value="0" Name="IsMotion"/>',
            )
        )
        self.assertTrue(
            listener._is_motion_event(
                "tns1:RuleEngine/CellMotionDetector/Motion",
                '<tt:SimpleItem Value="1" Name="IsMotion"/>',
            )
        )
        self.assertTrue(
            listener._is_motion_event(
                "tns1:RuleEngine/CellMotionDetector/Motion",
                (
                    '<tt:SimpleItem Value="0" Name="State"/>'
                    '<tt:SimpleItem Name="IsMotion" Value="true"/>'
                ),
            )
        )

    def test_onvif_structured_off_state_is_not_reported_as_motion(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())

        self.assertFalse(
            listener._is_motion_event(
                "tns1:RuleEngine/CellMotionDetector/Motion",
                "{'Name': 'IsMotion', 'Value': false}",
            )
        )
        self.assertIsNone(listener._motion_event_state("tns1:VideoSource", "signal ok"))

    def test_onvif_subscription_renews_before_expiration(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        manager = Mock()
        manager.Renew.return_value = SimpleNamespace(
            CurrentTime="2026-07-26T12:00:00Z",
            TerminationTime="2026-07-26T13:00:00Z",
        )
        listener._subscription_manager = manager
        listener.subscription_lifetime_seconds = 120
        listener._subscription_granted_lifetime_seconds = 3600
        listener._subscription_expires_monotonic = time.monotonic() + 120

        self.assertTrue(listener._subscription_renewal_due())
        self.assertTrue(listener._renew_subscription())

        manager.Renew.assert_called_once_with(TerminationTime="PT1H")
        self.assertEqual(listener.renewal_attempts, 1)
        self.assertEqual(listener.renewals, 1)
        self.assertEqual(listener.renewal_errors, 0)
        self.assertEqual(listener.subscription_lifetime_seconds, 3600)

    def test_onvif_poll_without_subscription_times_preserves_renewal_deadline(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        listener._record_subscription_times(SimpleNamespace(
            CurrentTime="2026-07-26T12:00:00Z",
            TerminationTime="2026-07-26T13:00:00Z",
        ))
        deadline = listener._subscription_expires_monotonic

        listener._record_subscription_times(SimpleNamespace())

        self.assertEqual(listener._subscription_expires_monotonic, deadline)
        self.assertEqual(listener.subscription_lifetime_seconds, 3600)

    def test_onvif_subscription_address_accepts_plain_string(self) -> None:
        subscription = SimpleNamespace(
            SubscriptionReference=SimpleNamespace(Address="http://camera/subscription")
        )

        self.assertEqual(
            OnvifEventListener._subscription_address(subscription),
            "http://camera/subscription",
        )


if __name__ == "__main__":
    unittest.main()
