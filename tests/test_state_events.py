from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
