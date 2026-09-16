"""Serializable per-camera spatial plan shared with the isolated native child.

Coordinates describe the uncropped live stream. No application dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math

ROI_LABEL = "__survng_inference_region__"


def spatial_plan(camera):
    zones = [zone.model_dump(mode="json") for zone in camera.zones]
    roi = camera.native_roi.model_dump(mode="json")
    revision = hashlib.sha256(json.dumps(zones, sort_keys=True).encode()).hexdigest()[:20]
    return {"zones": zones, "roi": roi, "revision": revision}


def analytics_zones(plan, width, height):
    return [{"id": str(index), "type": "polygon", "points": [
        {"x": round(p["x"] * width), "y": round(p["y"] * height)} for p in zone["points"]]}
        for index, zone in enumerate(plan.get("zones", []))
        if zone.get("enabled", True) and len(zone.get("points", [])) >= 3]


def inference_rectangle(plan, width, height):
    """One enclosing crop bounds cost to one inference per sampled frame.

    Padding is a fraction of the full frame, not the floor polygon's height.
    Periodic full-frame inference is supplied separately by the graph.
    """
    policy = plan.get("roi", {})
    names = set(policy.get("zone_names", []))
    points = [p for z in plan.get("zones", [])
              if z.get("enabled", True) and z.get("behavior", "incident") == "incident"
              and (not names or z["name"] in names) and len(z.get("points", [])) >= 3
              for p in z["points"]]
    if not policy.get("enabled") or not points:
        return (0, 0, width, height)
    padding = policy.get("padding", .15)
    left = max(0, math.floor((min(p["x"] for p in points) - padding) * width))
    top = max(0, math.floor((min(p["y"] for p in points) - padding) * height))
    right = min(width, math.ceil((max(p["x"] for p in points) + padding) * width))
    bottom = min(height, math.ceil((max(p["y"] for p in points) + padding) * height))
    return (left, top, max(1, right-left), max(1, bottom-top))


class RoiInput:
    """gvapython supplies a writable buffer header; pixel memory stays shared."""
    def __init__(self, plan, interval):
        self.plan, self.interval = plan, interval
        self.sequence = 0

    def process_frame(self, frame):
        info = frame.video_info()
        width, height = info.width, info.height
        fresh_index = self.sequence // self.interval
        period = self.plan["roi"].get("full_frame_interval", 5)
        rect = ((0, 0, width, height) if fresh_index % period == 0
                else inference_rectangle(self.plan, width, height))
        frame.add_region(*rect, ROI_LABEL, 1.0)
        self.sequence += 1
        return True
