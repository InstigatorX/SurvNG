from __future__ import annotations

import socket
import struct
import sys
import threading
import time
import unittest
from datetime import datetime, timezone
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
from survng.app.onvif_events import (
    OnvifEventListener,
    STOP_FORCE_SECONDS,
    STOP_GRACE_SECONDS,
)


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

    def test_onvif_response_arriving_after_stop_does_not_emit_callback(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        listener._stop.clear()
        notification = SimpleNamespace(Topic="motion", Message="motion")
        response = SimpleNamespace(NotificationMessage=[notification])
        pullpoint = Mock()

        def pull_messages(_request):
            listener._stop.set()
            return response

        pullpoint.PullMessages.side_effect = pull_messages
        callback = Mock()
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

        callback.assert_not_called()
        subscribe.assert_called_once()
        self.assertEqual(listener.poll_errors, 0)
        self.assertEqual(listener.notifications_received, 0)
        self.assertEqual(listener.motion_events_received, 0)
        self.assertEqual(listener.callback_errors, 0)

    def test_onvif_callback_failure_does_not_force_resubscription(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        listener._stop.clear()
        notification = SimpleNamespace(Topic="motion", Message="motion")
        pullpoint = Mock()
        pullpoint.PullMessages.return_value = SimpleNamespace(
            NotificationMessage=[notification]
        )

        def callback(*_args):
            listener._stop.set()
            raise RuntimeError("application failure")

        listener.on_motion = Mock(side_effect=callback)
        fake_onvif = type("FakeOnvifCamera", (), {})
        modules = {
            "onvif": SimpleNamespace(ONVIFCamera=fake_onvif),
            "zeep": SimpleNamespace(Transport=object),
            "zeep.cache": SimpleNamespace(SqliteCache=object),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(listener, "_subscribe", return_value=pullpoint),
            patch.object(listener, "_unsubscribe"),
            patch.object(listener, "_close_transport"),
        ):
            listener._run_until_stopped()

        listener.on_motion.assert_called_once()
        self.assertEqual(listener.poll_errors, 0)
        self.assertEqual(listener.notifications_received, 1)
        self.assertEqual(listener.motion_events_received, 1)
        self.assertEqual(listener.callback_errors, 1)

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
            side_effect=lambda _generation, stop: stop.wait(2),
        ):
            listener.start()
            listener.stop()

        self.assertEqual(order, ["unsubscribe", "close"])
        self.assertFalse(listener.running)

    def test_onvif_worker_is_daemonized_and_forced_stop_is_bounded(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        thread = Mock()
        thread.is_alive.return_value = True
        with patch("survng.app.onvif_events.threading.Thread", return_value=thread) as factory:
            listener.start()
            listener.stop()

        self.assertTrue(factory.call_args.kwargs["daemon"])
        self.assertEqual(
            [call.kwargs["timeout"] for call in thread.join.call_args_list],
            [STOP_GRACE_SECONDS, STOP_FORCE_SECONDS],
        )
        self.assertLessEqual(STOP_GRACE_SECONDS + STOP_FORCE_SECONDS, 15)

    def test_onvif_restart_waits_for_stopping_generation_before_replacing_it(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        stopping = Mock()
        stopping.is_alive.side_effect = [True, False, False]
        listener._thread = stopping
        listener._stop.set()
        replacement = Mock()

        with patch(
            "survng.app.onvif_events.threading.Thread",
            return_value=replacement,
        ):
            listener.start()

        stopping.join.assert_called_once_with(timeout=STOP_FORCE_SECONDS)
        replacement.start.assert_called_once_with()
        self.assertIs(listener._thread, replacement)

    def test_onvif_restart_rejects_a_generation_that_remains_stuck(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        stopping = Mock()
        stopping.is_alive.return_value = True
        listener._thread = stopping
        listener._stop.set()

        with self.assertRaisesRegex(RuntimeError, "still stopping"):
            listener.start()

        stopping.join.assert_called_once_with(timeout=STOP_FORCE_SECONDS)

    def test_onvif_start_cannot_cross_an_unfinished_stop_transaction(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        thread = Mock()
        thread.is_alive.return_value = True
        listener._thread = thread

        ticket = listener.request_stop()

        with self.assertRaisesRegex(RuntimeError, "stop is still pending"):
            listener.start()
        self.assertIs(listener.request_stop(), ticket)
        thread.start.assert_not_called()

    def test_stale_onvif_stop_ticket_cannot_finalize_a_new_generation(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        old_thread = Mock()
        old_thread.is_alive.return_value = False
        listener._thread = old_thread
        old_ticket = listener.request_stop()
        self.assertTrue(listener.wait_stopped(1.0, old_ticket))

        replacement = Mock()
        replacement.is_alive.return_value = True
        with patch(
            "survng.app.onvif_events.threading.Thread",
            return_value=replacement,
        ):
            listener.start()

        self.assertFalse(listener.wait_stopped(0.0, old_ticket))
        self.assertIs(listener._thread, replacement)
        self.assertTrue(listener.connected is False)

    def test_onvif_stop_during_subscribe_closes_generation_once(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        entered = threading.Event()
        released = threading.Event()
        transport = SimpleNamespace(session=Mock())
        transport.session.close.side_effect = released.set

        def subscribe(*_args, **_kwargs):
            with listener._lifecycle_lock:
                listener._transport = transport
                listener._transport_generation = listener._generation
            entered.set()
            self.assertTrue(released.wait(1))
            return Mock()

        with (
            patch.object(listener, "_subscribe", side_effect=subscribe),
            patch("survng.app.onvif_events.STOP_GRACE_SECONDS", 0.01),
            patch("survng.app.onvif_events.STOP_FORCE_SECONDS", 0.2),
        ):
            listener.start()
            self.assertTrue(entered.wait(1))
            listener.request_stop()
            self.assertTrue(listener.wait_stopped(1.0))

        transport.session.close.assert_called_once_with()
        listener.on_motion.assert_not_called()
        self.assertFalse(listener.running)

    def test_onvif_stop_during_pull_drops_returned_notification(self) -> None:
        callback = Mock()
        listener = OnvifEventListener(camera(onvif=True), callback)
        entered = threading.Event()
        released = threading.Event()
        manager = Mock()
        transport = SimpleNamespace(session=Mock())
        transport.session.close.side_effect = released.set
        notification = SimpleNamespace(Topic="motion", Message="motion")
        response = SimpleNamespace(NotificationMessage=[notification])
        pullpoint = Mock()

        def pull_messages(_request):
            entered.set()
            self.assertTrue(released.wait(1))
            return response

        pullpoint.PullMessages.side_effect = pull_messages

        def subscribe(*_args, **_kwargs):
            with listener._lifecycle_lock:
                generation = listener._generation
                listener._transport = transport
                listener._transport_generation = generation
                listener._subscription_manager = manager
                listener._subscription_generation = generation
            return pullpoint

        with (
            patch.object(listener, "_subscribe", side_effect=subscribe),
            patch("survng.app.onvif_events.STOP_GRACE_SECONDS", 0.01),
            patch("survng.app.onvif_events.STOP_FORCE_SECONDS", 0.2),
        ):
            listener.start()
            self.assertTrue(entered.wait(1))
            listener.request_stop()
            self.assertTrue(listener.wait_stopped(1.0))

        callback.assert_not_called()
        manager.Unsubscribe.assert_called_once_with()
        transport.session.close.assert_called_once_with()
        self.assertEqual(listener.notifications_received, 0)
        self.assertFalse(listener.running)

    def test_onvif_stop_during_renew_cannot_resume_polling(self) -> None:
        callback = Mock()
        listener = OnvifEventListener(camera(onvif=True), callback)
        entered = threading.Event()
        released = threading.Event()
        manager = Mock()
        transport = SimpleNamespace(session=Mock())
        transport.session.close.side_effect = released.set
        pullpoint = Mock()

        def renew(**_kwargs):
            entered.set()
            self.assertTrue(released.wait(1))
            return SimpleNamespace(
                CurrentTime="2026-08-08T12:00:00Z",
                TerminationTime="2026-08-08T13:00:00Z",
            )

        manager.Renew.side_effect = renew

        def subscribe(*_args, **_kwargs):
            with listener._lifecycle_lock:
                generation = listener._generation
                listener._transport = transport
                listener._transport_generation = generation
                listener._subscription_manager = manager
                listener._subscription_generation = generation
            listener._subscription_granted_lifetime_seconds = 60.0
            listener._subscription_expires_monotonic = time.monotonic()
            return pullpoint

        with (
            patch.object(listener, "_subscribe", side_effect=subscribe),
            patch("survng.app.onvif_events.STOP_GRACE_SECONDS", 0.01),
            patch("survng.app.onvif_events.STOP_FORCE_SECONDS", 0.2),
        ):
            listener.start()
            self.assertTrue(entered.wait(1))
            listener.request_stop()
            self.assertTrue(listener.wait_stopped(1.0))

        manager.Renew.assert_called_once_with(TerminationTime="PT1H")
        manager.Unsubscribe.assert_called_once_with()
        transport.session.close.assert_called_once_with()
        pullpoint.PullMessages.assert_not_called()
        callback.assert_not_called()
        self.assertEqual(listener.renewals, 0)
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

    def test_onvif_effectiveness_degrades_without_changing_transport_health(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        listener.connected = True
        listener.notifications_received = 10
        listener.motion_events_received = 2
        listener.inactive_motion_events = 6
        now = datetime.now(timezone.utc)

        for _ in range(3):
            listener.record_ema_observation(False, now)

        status = listener.effectiveness_snapshot()
        self.assertTrue(listener.connected)
        self.assertEqual(status["signal_effectiveness_status"], "degraded")
        self.assertTrue(status["signal_degraded"])
        self.assertEqual(status["ema_window_observations"], 3)
        self.assertEqual(status["ema_window_without_onvif"], 3)
        self.assertEqual(status["ema_window_match_rate"], 0.0)
        self.assertEqual(status["recognized_notifications"], 8)
        self.assertEqual(status["notification_recognition_rate"], 0.8)
        self.assertEqual(status["active_motion_rate"], 0.25)

    def test_onvif_effectiveness_requires_enough_ema_evidence(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        listener.connected = True

        listener.record_ema_observation(False)
        listener.record_ema_observation(False)

        status = listener.effectiveness_snapshot()
        self.assertEqual(status["signal_effectiveness_status"], "insufficient_data")
        self.assertFalse(status["signal_degraded"])

    def test_unknown_notification_samples_are_bounded_and_do_not_store_payload(self) -> None:
        listener = OnvifEventListener(camera(onvif=True), Mock())
        observed = datetime.now(timezone.utc)

        for index in range(8):
            listener._record_unknown_notification(
                f"tns1:Device/Status/{index}",
                f"username=admin&password=secret-{index}",
                observed,
            )

        samples = listener.effectiveness_snapshot()["unknown_notification_samples"]
        self.assertEqual(len(samples), 5)
        self.assertEqual(samples[0]["topic"], "tns1:Device/Status/3")
        self.assertIn("message_fingerprint", samples[0])
        self.assertNotIn("message", samples[0])
        self.assertNotIn("secret", repr(samples))

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
