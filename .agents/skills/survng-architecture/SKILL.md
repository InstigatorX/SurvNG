---
name: survng-architecture
description: Trace or design SurvNG component boundaries, state ownership, worker lifecycles, APIs, and data flow before a substantial change or for a read-only architecture investigation.
---

# Trace or design SurvNG architecture

1. Use SurvNG research to follow one real path from entry point to result.
   Identify owners of state, resources, and side effects. Trace startup,
   rollback, cancellation, reconfiguration, shutdown, and overlapping
   requests where relevant.
2. Write down caller contracts and invariants at each boundary, including
   ordering, generation identity, lock ownership, persistence, recovery, and
   compatibility. Check corresponding tests and locate coverage gaps.
3. For design work, compare concrete alternatives with proposed file and API
   boundaries and migration order. Prefer clear ownership and the smallest
   scope that fits the constraints. For a contested decision, ask independent
   agents to assess the options against the same criteria when available.
   A scoped probe may resolve an empirical unknown if authorized.
4. Identify an observable check for each invariant the change can affect.
   Cite entry points, state transitions, tests, or evidence gaps. For a
   read-only request, deliver the architecture and verification findings.
