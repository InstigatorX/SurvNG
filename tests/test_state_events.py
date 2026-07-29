from __future__ import annotations

import unittest
import threading
from unittest.mock import patch

from survng.app.state_events import StateEventBroker


class StateEventBrokerTest(unittest.TestCase):
    def test_publishes_typed_ordered_events(self) -> None:
        broker = StateEventBroker()
        subscriber = broker.subscribe()

        first = broker.publish("camera_state", {"id": "gate"})
        second = broker.publish("incident", {"event_id": 42})

        self.assertEqual(subscriber.get_nowait(), first)
        self.assertEqual(subscriber.get_nowait(), second)
        self.assertEqual(int(first.id.rsplit(":", 1)[1]) + 1, int(second.id.rsplit(":", 1)[1]))

    def test_slow_subscriber_keeps_newest_event(self) -> None:
        broker = StateEventBroker(subscriber_queue_size=8)
        subscriber = broker.subscribe()
        for index in range(12):
            broker.publish("camera_state", {"index": index})

        values = []
        while not subscriber.empty():
            values.append(subscriber.get_nowait().data["index"])

        self.assertEqual(values[-1], 11)
        self.assertNotIn(0, values)

    def test_close_disconnects_subscribers(self) -> None:
        broker = StateEventBroker()
        subscriber = broker.subscribe()
        broker.close()

        self.assertIsNone(subscriber.get_nowait())
        self.assertIsNone(broker.publish("camera_state", {}))

    def test_replays_events_after_cursor(self) -> None:
        broker = StateEventBroker()
        first = broker.publish("camera_state", {"enabled": False})
        second = broker.publish("camera_state", {"enabled": True})

        self.assertEqual(broker.events_after(first.id), [second])
        self.assertIsNone(broker.events_after("another-boot:1"))

    def test_rejects_impossible_future_and_negative_cursors(self) -> None:
        broker = StateEventBroker()
        broker.publish("camera_state", {"enabled": True})

        self.assertIsNone(broker.events_after(f"{broker.instance_id}:999"))
        self.assertIsNone(broker.events_after(f"{broker.instance_id}:-1"))
        self.assertEqual(broker.sequence(f"{broker.instance_id}:1"), 1)
        self.assertIsNone(broker.sequence("another-boot:1"))

    def test_published_data_is_an_immutable_snapshot(self) -> None:
        broker = StateEventBroker()
        payload = {"camera": {"enabled": True}}

        event = broker.publish("camera_state", payload)
        payload["camera"]["enabled"] = False

        self.assertTrue(event.data["camera"]["enabled"])

    def test_close_sentinel_cannot_overtake_accepted_publish(self) -> None:
        broker = StateEventBroker()
        subscriber = broker.subscribe()
        entered_delivery = threading.Event()
        allow_delivery = threading.Event()
        original_put = subscriber.put_nowait

        def blocking_put(item):
            if item is not None:
                entered_delivery.set()
                allow_delivery.wait(timeout=1)
            original_put(item)

        with patch.object(subscriber, "put_nowait", side_effect=blocking_put):
            publisher = threading.Thread(target=broker.publish, args=("camera_state", {"id": "gate"}))
            publisher.start()
            self.assertTrue(entered_delivery.wait(timeout=1))
            closer = threading.Thread(target=broker.close)
            closer.start()
            self.assertTrue(closer.is_alive())
            allow_delivery.set()
            publisher.join(timeout=1)
            closer.join(timeout=1)

        self.assertEqual(subscriber.get_nowait().type, "camera_state")
        self.assertIsNone(subscriber.get_nowait())

    def test_slow_payload_copy_does_not_block_broker_close(self) -> None:
        broker = StateEventBroker()
        copy_started = threading.Event()
        allow_copy = threading.Event()

        class SlowPayload:
            def __deepcopy__(self, _memo):
                copy_started.set()
                allow_copy.wait(timeout=1)
                return {"copied": True}

        publisher = threading.Thread(
            target=broker.publish,
            args=("camera_state", SlowPayload()),
        )
        publisher.start()
        self.assertTrue(copy_started.wait(timeout=1))

        closer = threading.Thread(target=broker.close)
        closer.start()
        closer.join(timeout=0.2)
        self.assertFalse(closer.is_alive())

        allow_copy.set()
        publisher.join(timeout=1)
        self.assertFalse(publisher.is_alive())

    def test_publish_after_close_does_not_copy_payload(self) -> None:
        broker = StateEventBroker()
        broker.close()

        class UncopyablePayload:
            def __deepcopy__(self, _memo):
                raise AssertionError("closed broker must not copy payload")

        self.assertIsNone(broker.publish("camera_state", UncopyablePayload()))


if __name__ == "__main__":
    unittest.main()
