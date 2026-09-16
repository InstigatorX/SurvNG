"""Continuous polygon geometry for motion metadata; no video pixels are copied."""
from __future__ import annotations


def _hull(points):
    points = sorted(set(points))

    def half(sequence):
        result = []
        for p in sequence:
            while len(result) >= 2:
                a, b = result[-2:]
                if (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) > 0:
                    break
                result.pop()
            result.append(p)
        return result

    return half(points)[:-1] + half(reversed(points))[:-1]


def _edges(polygons):
    return [(a, b) for polygon in polygons for a, b in zip(polygon, polygon[1:] + polygon[:1])]


def _intervals(polygons, y, left, right):
    intervals = []
    for polygon in polygons:
        xs = sorted(a[0] + (y-a[1])*(b[0]-a[0])/(b[1]-a[1])
                    for a, b in _edges([polygon]) if (a[1] > y) != (b[1] > y))
        intervals.extend((max(left, a), min(right, b)) for a, b in zip(xs[::2], xs[1::2])
                         if max(left, a) < min(right, b))
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


class MotionRegions:
    """Test whether a rectangle has positive eligible area outside exclusions.

    Split at polygon vertices and edge intersections. Between these heights,
    edge ordering is fixed, so one horizontal slice detects any remaining area.
    This supports concave polygons and the union of overlapping exclusions.
    """

    def __init__(self, incident_zones, exclusion_zones, padding):
        self.allowed = []
        for zone in incident_zones:
            polygon = [(p['x'], p['y']) for p in zone['points']]
            self.allowed.append(polygon)
            if padding:
                # Square dilation matches the budget's existing rectangle padding.
                for a, b in _edges([polygon]):
                    self.allowed.append(_hull([(p[0]+dx, p[1]+dy) for p in (a, b)
                                              for dx in (-padding, padding)
                                              for dy in (-padding, padding)]))
        if not incident_zones:
            self.allowed = [[(0, 0), (1, 0), (1, 1), (0, 1)]]
        self.excluded = [[(p['x'], p['y']) for p in zone['points']] for zone in exclusion_zones]
        self.edges = _edges(self.allowed + self.excluded)
        cuts = {p[1] for edge in self.edges for p in edge}
        for i, (a, b) in enumerate(self.edges):
            dx, dy = b[0]-a[0], b[1]-a[1]
            for c, d in self.edges[i+1:]:
                ex, ey = d[0]-c[0], d[1]-c[1]
                determinant = dx*ey-dy*ex
                if abs(determinant) < 1e-15:
                    continue
                t = ((c[0]-a[0])*ey-(c[1]-a[1])*ex)/determinant
                u = ((c[0]-a[0])*dy-(c[1]-a[1])*dx)/determinant
                if 0 < t < 1 and 0 < u < 1:
                    cuts.add(a[1]+t*dy)
        self.cuts = sorted(cuts)

    def allows(self, rect):
        left, top, right, bottom = rect
        left, top, right, bottom = max(0, left), max(0, top), min(1, right), min(1, bottom)
        if left >= right or top >= bottom:
            return False
        cuts = {top, bottom, *(y for y in self.cuts if top < y < bottom)}
        for a, b in self.edges:
            if a[0] == b[0]:
                continue
            for x in (left, right):
                t = (x-a[0])/(b[0]-a[0])
                y = a[1]+t*(b[1]-a[1])
                if 0 < t < 1 and top < y < bottom:
                    cuts.add(y)
        cuts = sorted(cuts)
        for top, bottom in zip(cuts, cuts[1:]):
            y = (top+bottom)/2
            excluded = _intervals(self.excluded, y, left, right)
            for start, end in _intervals(self.allowed, y, left, right):
                for a, b in excluded:
                    if b <= start:
                        continue
                    if a > start:
                        break
                    start = max(start, b)
                if end-start > 1e-12:
                    return True
        return False
