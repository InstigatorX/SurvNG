---
name: survng-orchestration
description: Coordinate substantial SurvNG work with bounded research, isolated implementation, targeted testing, and independent code review. Use when work has meaningful ownership boundaries or the user requests delegation.
---

# Orchestrate SurvNG work

1. Read the applicable AGENTS.md and capture the base commit, working-tree
   state, desired outcome, permissions, and observable acceptance checks.
   Mark investigation requests read-only.
2. Split work by dependencies. Assign read-only research and test discovery
   concurrently when independent. Resolve research that affects the design
   or acceptance checks before assigning implementation.
3. Brief each delegate with the relevant snapshot (including any pending
   changes), question, permitted actions, owned files, and expected evidence.
   Ask for paths or commands, results, and remaining uncertainties. Identify
   read-only tasks explicitly even when the harness lacks read-only agents.
4. Give concurrent writers separate worktrees from a known base and disjoint
   ownership. Include pending prerequisites in each worker's snapshot
   while preserving the original checkout. If isolation cannot be created,
   serialize writes. Keep test artifacts outside the checkout where practical
   and inspect each patch before incorporating it.
5. Once the candidate is ready, give a verifier and an independent reviewer
   the same snapshot. They can run concurrently if each has isolated runtime
   state or uses static checks. Resolve supported findings, rerun affected
   checks against the integrated snapshot, and check the final diff.
6. Report the result, who did what, verification evidence, and open issues.
   If subagents are unavailable, perform the stages locally and disclose
   that an independent agent review was unavailable. Get approval before
   merging. Follow the user's scope for publication and other remote actions.
