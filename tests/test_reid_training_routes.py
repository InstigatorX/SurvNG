from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import TestCase

from survng.app.manager_access import ManagerAccessCoordinator
from survng.app.reid_training_routes import (
    ReidTrainingRouteDependencies,
    create_reid_training_router,
)


class ReidTrainingRouteTests(TestCase):
    def test_status_uses_manager_generation_lock_before_manager_getter(self) -> None:
        manager = SimpleNamespace(
            config=SimpleNamespace(
                detector=SimpleNamespace(
                    tracking=SimpleNamespace(reid_training_collector_enabled=True)
                )
            ),
            reid_training=SimpleNamespace(status=lambda: {"samples": 7}),
        )
        bundle = create_reid_training_router(
            ReidTrainingRouteDependencies(
                get_manager=lambda: manager,
                manager_lock=threading.RLock(),
                manager_access=ManagerAccessCoordinator(),
            )
        )

        self.assertEqual(
            bundle.handlers["reid_training_status"](),
            {"collector_enabled": True, "samples": 7},
        )
