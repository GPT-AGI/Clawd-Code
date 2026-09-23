# Teammate Worker Lifecycle

## Current control model

The lead owns worker lifecycle changes. Teammates cannot stop or resume one
another, change another task, or invoke team-management tools.

Worker states:

```text
created -> running -> idle -> completed
              |        |
              +-> stopping -> cancelled -> running
              +-> failed -----------------> running
```

`TeammateStop` records a stop request before changing tasks. A running worker
observes that request at the next model or tool boundary. The task policy is:

- `requeue`: return unfinished work to `pending` with no owner.
- `cancel`: mark unfinished work `cancelled`.

Other workers continue. The team stays `running` and may return `blocked` when
the stopped worker leaves an unassigned or cancelled dependency. The lead or a
human operator can then resume a worker, reassign the task, and resume the team.

Every request and acknowledgement is auditable through `agent.stop_requested`,
`run.cancelled`, `agent.stopped`, `task.requeued`, `task.cancelled`,
`agent.resumed`, and `task.reassigned` trace events.

## Why force is not exposed yet

Workers currently execute in a `ThreadPoolExecutor`. Python cannot safely kill
one running thread. A graceful stop can prevent the next model or tool call, but
it cannot interrupt an HTTP request or a shell command already in progress.
Calling that behavior `force` would provide a false guarantee.

## Process-worker migration

True force termination requires the following architecture:

1. Run each worker in its own process and process group.
2. Recreate provider and tool dependencies inside the child instead of sharing
   non-picklable clients from the lead process.
3. Send commands over a small supervisor channel and keep team/task/events in
   the existing durable store.
4. Persist worker PID, process start identity, heartbeat, and supervisor epoch
   so stale PIDs cannot be terminated accidentally after a restart.
5. Implement `graceful` as a stop request followed by a bounded grace period.
6. Implement `force` as process-group termination after that grace period,
   including descendant shell processes.
7. Recover the task lease using the same `requeue` or `cancel` policy after the
   child exits.
8. Use `terminate()`/`kill()` on POSIX and the corresponding Windows process
   APIs behind one supervisor abstraction.

## Force-mode acceptance criteria

- Stopping one worker never changes another worker or the team cancellation flag.
- A force-stopped worker cannot emit tool results after acknowledgement.
- Child shell processes are gone before the task lease is released.
- Repeated stop requests are idempotent.
- Supervisor restart cannot kill an unrelated process with a reused PID.
- Trace order shows request, signal, process exit, task disposition, and final
  worker state.
- Linux, macOS, and Windows integration tests cover graceful and force modes.
