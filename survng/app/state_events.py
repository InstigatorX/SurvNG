from __future__ import annotations

import queue
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StateEvent:
    id: str
    type: str
    data: Any


class StateEventBroker:
    def __init__(self, subscriber_queue_size: int = 256) -> None:
        self._subscriber_queue_size = max(8, int(subscriber_queue_size))
        self._subscribers: set[queue.Queue[StateEvent | None]] = set()
        self._history: deque[StateEvent] = deque(maxlen=512)
        self._lock = threading.Lock()
        self._next_id = 1
        self.instance_id = uuid.uuid4().hex[:12]
        self._closed = False

    @property
    def cursor(self) -> str:
        with self._lock:
            return f"{self.instance_id}:{self._next_id - 1}"

    def subscribe(self) -> queue.Queue[StateEvent | None]:
        subscriber: queue.Queue[StateEvent | None] = queue.Queue(maxsize=self._subscriber_queue_size)
        with self._lock:
            if self._closed:
                subscriber.put_nowait(None)
            else:
                self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[StateEvent | None]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)

    def publish(self, event_type: str, data: Any) -> StateEvent | None:
        with self._lock:
            if self._closed:
                return None
            event = StateEvent(f"{self.instance_id}:{self._next_id}", str(event_type), data)
            self._next_id += 1
            self._history.append(event)
            subscribers = tuple(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    pass
        return event

    def events_after(self, event_id: str) -> list[StateEvent] | None:
        prefix = f"{self.instance_id}:"
        if not str(event_id).startswith(prefix):
            return None
        try:
            sequence = int(str(event_id)[len(prefix):])
        except ValueError:
            return None
        with self._lock:
            history = list(self._history)
            if history:
                oldest = int(history[0].id.rsplit(":", 1)[1])
                if sequence < oldest - 1:
                    return None
            return [event for event in history if int(event.id.rsplit(":", 1)[1]) > sequence]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscribers = tuple(self._subscribers)
            self._subscribers.clear()
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(None)
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
