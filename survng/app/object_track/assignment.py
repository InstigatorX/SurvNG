"""Dependency-free one-to-one assignment for offline tracking evaluation."""

from __future__ import annotations

import math


def maximum_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Maximize total weight, leaving zero-weight edges unmatched.

    Accept a rectangular matrix of finite, nonnegative weights. Rows and columns
    are each used at most once. Equal-cost decisions follow input index order so
    repeated runs are deterministic. Empty matrices return no assignments.

    This is the rectangular Hungarian algorithm with one zero-weight dummy
    column per row. Dummy columns allow any row to remain unmatched. Complexity
    is O(rows**2 * (columns + rows)); no numerical/scientific package is needed.
    """
    if not weights:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if any(len(row) != column_count for row in weights):
        raise ValueError("assignment weights must be rectangular")
    if any(not math.isfinite(value) or value < 0 for row in weights for value in row):
        raise ValueError("assignment weights must be finite and nonnegative")
    if not column_count or not any(value > 0 for row in weights for value in row):
        return []

    # One-based potentials, column ownership, and augmenting-path predecessors.
    total_columns = column_count + row_count
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (total_columns + 1)
    owner = [0] * (total_columns + 1)
    predecessor = [0] * (total_columns + 1)
    for row_index in range(1, row_count + 1):
        owner[0] = row_index
        column = 0
        distance = [math.inf] * (total_columns + 1)
        visited = [False] * (total_columns + 1)
        while True:
            visited[column] = True
            current_row = owner[column]
            delta = math.inf
            next_column = 0
            for candidate in range(1, total_columns + 1):
                if visited[candidate]:
                    continue
                cost = (
                    -weights[current_row - 1][candidate - 1]
                    if candidate <= column_count else 0.0
                )
                reduced_cost = (
                    cost - row_potential[current_row] - column_potential[candidate]
                )
                if reduced_cost < distance[candidate]:
                    distance[candidate] = reduced_cost
                    predecessor[candidate] = column
                if distance[candidate] < delta:
                    delta = distance[candidate]
                    next_column = candidate
            for candidate in range(total_columns + 1):
                if visited[candidate]:
                    row_potential[owner[candidate]] += delta
                    column_potential[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            previous = predecessor[column]
            owner[column] = owner[previous]
            column = previous

    return sorted(
        (owner[column] - 1, column - 1)
        for column in range(1, column_count + 1)
        if owner[column] and weights[owner[column] - 1][column - 1] > 0
    )
