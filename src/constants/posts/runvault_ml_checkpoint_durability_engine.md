# Why Most ML Training Recovery Is Broken: Building RunVault's Zero-Dependency Checkpoint Durability Engine

*Full source code, architecture decision records, unit tests, and the published package are available on GitHub: [Black-fox17/run-vault](https://github.com/Black-fox17/run-vault) and PyPI: [runvault-ai](https://pypi.org/project/runvault-ai/).*

---

If you train deep learning models on spot instances, cloud VMs, or Google Colab, you know the sinking feeling: you kick off an 8-hour training run overnight, come back in the morning, and the runtime has disconnected. You try to resume from `checkpoint_best.pth`, and PyTorch either crashes with an obscure `RuntimeError: Error(s) in loading state_dict`, or worse—it silently loads corrupted, half-written weights and continues training straight into `NaN` loss.

Why is checkpointing in modern machine learning so astonishingly fragile?

We have massive, multi-million-dollar experiment tracking platforms (Weights & Biases, MLflow, Neptune). But when you inspect what actually happens under the hood, almost all of them treat model checkpoints as an afterthought: a secondary file blob uploaded after the fact over an HTTP request, wrapped in bloated server daemons that mandate dozens of heavy third-party dependencies inside your training environment.

When we set out to build **RunVault (`runvault-ai`)**, our objective was to treat checkpoint durability and experiment recovery as a first-class operating systems problem, not a loose Python script.

We wanted an offline-first, lightweight SDK that an ML engineer could drop into any PyTorch loop with zero configuration. It had to guarantee byte-for-byte durability against hard process crashes, stream multi-gigabyte models directly to storage without server bottlenecks, and recover atomically even when underlying cloud drives timeout mid-upload.

Here is the honest, start-to-finish breakdown of what broke in standard ML tooling, the architectural traps we threw away, and how we engineered RunVault from scratch.

---

## Surprise #1: The Bloated Daemon Anti-Pattern (And the Library-First Invariant)

When you look at popular experiment management tools, the standard deployment model looks like this:
1. Spin up a Docker container running a background tracking daemon.
2. Install a 150MB Python SDK pulling in FastAPI, Pydantic, SQLAlchemy, gRPC, and heavy cloud SDKs directly into your PyTorch training environment.
3. Hope that none of those packages create dependency conflicts with your CUDA runtime or NumPy versions.

If you are training on an ephemeral Google Colab notebook or a remote HPC cluster without root access, this model falls apart immediately. If the background daemon dies, your training loop freezes or crashes.

We established our first non-negotiable architectural rule:

> **The Library-First Invariant:** The core RunVault SDK must operate completely independently, offline, and without requiring any server, daemon, or external service.

The core `runvault` package has **zero mandatory third-party dependencies**. It relies strictly on Python's standard library (`sqlite3`, `hashlib`, `os`, `typing`).

```text
┌────────────────────────────────────────────────────────┐
│                   User Training Loop                   │
│          (PyTorch / TensorFlow / JAX / NumPy)          │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│              RunVault Core SDK (runvault-ai)           │
│        Zero Third-Party Dependencies (Stdlib-Only)     │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
┌──────────────▼─────────────┐   ┌────────▼──────────────┐
│  ACID Metadata Storage     │   │  Direct-to-Storage    │
│  Embedded SQLite (.db)     │   │  Data Plane (Local /  │
│  PRAGMA synchronous = FULL │   │  Google Drive Stream) │
└────────────────────────────┘   └───────────────────────┘
```

Heavy integrations like PyTorch (`run-vault[torch]`) and Google Drive API (`run-vault[gdrive]`) are isolated optional extras. Importing the core `runvault` library takes **under 2 milliseconds** and never touches `torch` or `google.auth` unless those specific integrations are invoked.

---

## Surprise #2: The Lie of `torch.save(model.state_dict(), path)`

Most ML tutorials tell you to save checkpoints like this:

```python
# The standard textbook approach
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pth')
```

What actually happens if your spot instance gets preempted or someone kills the process while `torch.save` is running?
- Python opens `checkpoint.pth` with write mode (`wb`), truncating the existing file to 0 bytes.
- The operating system writes serialized tensor buffers into the OS page cache.
- The machine loses power before the page cache flushes to physical non-volatile storage.

The result is a **corrupted, 0-byte or partially written file**. Your latest checkpoint is destroyed, and your previous checkpoint was already overwritten.

### The POSIX Atomic Publication Protocol

In RunVault, checkpoints are never written directly to their destination path. The persistence lifecycle follows strict operating-system-level guarantees:

| Stage | Mechanism | Operating System Guarantee |
|---|---|---|
| **1. Staging** | Write to `<run_id>_<uuid>.tmp` | Isolated write; existing checkpoints cannot be touched. |
| **2. Buffer Flush** | `os.fdatasync(fd)` (or `os.fsync`) | Forces dirty kernel page buffers to physical non-volatile disk. |
| **3. Directory Sync** | `os.fsync(dir_fd)` (POSIX) | Commits the containing directory's inode metadata to disk. |
| **4. Atomic Publish** | `os.link(staged, final)` | Atomic POSIX hard link; guarantees no-overwrite collision prevention. |
| **5. Cleanup** | `os.unlink(staged)` | Removes staging reference cleanly. |

If anything crashes at step 1, 2, or 3, the destination checkpoint file **never exists**. The previous valid checkpoint remains 100% intact.

```python
# How RunVault enforces physical durability before publication
dest_fd = os.open(staging_path, os.O_WRONLY)
try:
    if hasattr(os, "fdatasync"):
        os.fdatasync(dest_fd)  # Flushes file payload without updating access timestamps
    else:
        os.fsync(dest_fd)
finally:
    os.close(dest_fd)

# Atomic link ensures the checkpoint appears instantaneously or not at all
os.link(staging_path, final_published_path)
os.unlink(staging_path)
```

### The SQLite Durability Ordering Trap

Tracking checkpoint metadata (run ID, step, epoch, SHA-256 hash, metrics) requires database ACID properties. But SQLite's default mode (`PRAGMA synchronous = NORMAL`) allows transactions to return before the database file is physically flushed to disk.

To guarantee durability, RunVault executes `PRAGMA synchronous = FULL` on every checkpoint transaction. But here is the trap:

> In SQLite, executing `PRAGMA synchronous = FULL` **inside** an already-active `BEGIN IMMEDIATE` transaction is silently ignored!

To guarantee physical durability without race conditions, RunVault executes:
1. `PRAGMA synchronous = FULL` on the dedicated connection **before** beginning the transaction.
2. `BEGIN IMMEDIATE` to acquire an exclusive writer lock.
3. Insert checkpoint metadata with exact SHA-256 digest and file size.
4. `COMMIT` the transaction physically to disk.
5. Revert the connection policy to `NORMAL` for lightweight metric logging.

---

## Surprise #3: The Server Checkpoint Bottleneck (Direct-to-Storage Cloud Streaming)

In distributed and cloud training setups, routing multi-gigabyte model weights through an API server is an architectural anti-pattern:

```text
Standard MLflow Architecture (Bottleneck):
GPU Worker  ──[ 4 GB binary ]──►  API Gateway Server  ──►  S3 / GCS Storage
(Worker stalls waiting for API server network queue)

RunVault Direct-to-Storage Architecture:
GPU Worker  ──[ Metadata JSON ]───────────────────────►  Local SQLite / DB
            ──[ 4 GB binary chunk stream ]────────────►  Remote Storage (Google Drive)
(Zero binary payload overhead on metadata systems)
```

In Google Colab, free GPU runtimes disconnect after idle timeouts or 12-hour limits. If your checkpoints stay on Colab's local disk, they vanish forever when the runtime terminates.

We implemented RunVault's first remote storage backend: **Google Drive (`runvault-ai[gdrive]`)**.

### Lost-Response Upload Idempotency

When uploading 2GB checkpoints over Google Drive's chunked resumable media API, a classic distributed systems problem occurs:
- The client uploads the final chunk.
- Google Drive receives the data and commits the file.
- The client's HTTP connection drops before receiving the `200 OK` response.

If a naive client retries, it uploads the file again, generating **duplicate orphan files** in Drive and wasting bandwidth.

RunVault solves this by computing deterministic metadata before transmission:
1. Every checkpoint is tagged with a unique `appProperties.runvault_checkpoint_id` and SHA-256 hash.
2. On any retry or connection timeout, RunVault issues a preflight metadata query searching for that specific checkpoint ID in the target folder.
3. If Drive reports that the file was already committed with matching hash and size, RunVault safely recovers the uploaded artifact without sending a single redundant byte.

### Bounded-Memory Download Streaming

Restoring a 5GB checkpoint shouldn't consume 5GB of host RAM just to deserialize tensors. RunVault's `GoogleDriveDownloadStream` implements bounded-memory chunk swapping: it fetches chunks via `MediaIoBaseDownload` into a small sliding buffer, satisfying the full Python binary I/O protocol (`read`, `seekable`, `readinto`) in $O(\text{chunk\_size})$ memory.

---

## Surprise #4: PyTorch State Restoration Is NOT Atomic

Here was the most dangerous defect we discovered in standard PyTorch training pipelines:

```python
# The standard naive resume loop:
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
```

What happens if `model.load_state_dict()` succeeds, but `optimizer.load_state_dict()` crashes with `ValueError: loaded state dict has a different number of parameter groups`?

In standard PyTorch, **your model weights have already been mutated in place!** Your live model in memory now holds the loaded weights, but your optimizer still holds the old learning rate and momentum buffers. If you catch the exception and fall back to training, your model and optimizer are in an inconsistent, split-brain state.

### The Snapshot & Rollback Protocol (P0 Recovery Atomicity)

To guarantee that live training components are never left partially mutated, RunVault's `restore_torch_checkpoint()` executes an atomic snapshot and rollback protocol:

```python
# 1. Deep snapshot in-memory states before touching live objects
model_snapshot = {k: v.clone() for k, v in model.state_dict().items()}
optimizer_snapshot = copy.deepcopy(optimizer.state_dict()) if optimizer else None
scheduler_snapshot = copy.deepcopy(scheduler.state_dict()) if scheduler else None
scaler_snapshot = copy.deepcopy(scaler.state_dict()) if scaler else None
rng_snapshot = capture_rng_state() if restore_rng else None

try:
    # 2. Mutate components in strict sequence
    model.load_state_dict(payload["model_state_dict"], strict=strict_model)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    # 3. Restore RNG strictly LAST so tensor loads consume zero entropy
    if restore_rng and payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])
except Exception as err:
    # 4. Atomic Rollback: Restore all objects to pre-call snapshots on failure
    _rollback_components(
        model=model, model_snapshot=model_snapshot,
        optimizer=optimizer, optimizer_snapshot=optimizer_snapshot,
        scheduler=scheduler, scheduler_snapshot=scheduler_snapshot,
        scaler=scaler, scaler_snapshot=scaler_snapshot,
        restore_rng=restore_rng, rng_snapshot=rng_snapshot,
    )
    raise err
```

If any single component fails during restoration, the model, optimizer, scheduler, and RNGs are restored byte-for-byte to their original values before the error propagates.

---

## Surprise #5: The PyTorch `weights_only=True` Security Boundary

In PyTorch 2.0+, `torch.load()` introduced `weights_only=True` to prevent arbitrary code execution vulnerabilities caused by malicious unpickling.

When you save custom Python objects (such as NumPy RNG states):
`np.random.get_state()` returns a tuple containing a `numpy.ndarray` with internal uint32 keys. When deserialized with `weights_only=True`, PyTorch's safe unpickler rejects NumPy internal array classes (`numpy._core.multiarray._reconstruct`), crashing with `pickle.UnpicklingError`.

Instead of lazily setting `weights_only=False` (which exposes users to remote code execution from untrusted checkpoints), RunVault decomposes NumPy's RNG state into **primitive Python integer lists and scalars** during serialization:

```python
# Extract NumPy uint32 array elements as standard Python ints
np_state = np.random.get_state(legacy=True)
raw_tuple = tuple(np_state)
keys_list = [int(x) for x in raw_tuple[1]]  # Safe Python int primitives!

state["numpy"] = (
    str(raw_tuple[0]),
    keys_list,
    int(raw_tuple[2]),
    int(raw_tuple[3]),
    float(raw_tuple[4]),
)
```

During restoration, RunVault reconstructs the `np.uint32` array on the fly. This guarantees full bitwise NumPy reproducibility while maintaining **`weights_only=True` safe loading by default**.

---

## Test Suite & Verification Rigor

We enforced a strict verification bar before shipping:
- **155 deterministic tests** across SQLite ACID durability, concurrent worker contention, Google Drive mock streaming, and PyTorch rollback failures.
- **Strict-mode Pyright** (`typeCheckingMode = "strict"`) with **0 errors** across all modules.
- **Transparent older-checkpoint fallback:** If a checkpoint's SHA-256 hash does not match (detecting bit rot or truncated files), RunVault logs a warning and automatically falls back to the newest valid older checkpoint.

```bash
# Our mandatory verification pipeline
.venv/bin/ruff check .           # All checks passed!
.venv/bin/ruff format --check .  # 49 files left unchanged
.venv/bin/pyright                # 0 errors, 0 warnings, 0 informations
.venv/bin/pytest -v              # 155 passed in 18.91s
.venv/bin/uv build               # Successfully built runvault_ai-0.1.0-py3-none-any.whl
```

---

## What We Learned & What Comes Next

Building RunVault reinforced a fundamental lesson: **ML experiment tracking should be a library, not a server.**

By treating durability as a systems problem—atomic POSIX hard-links, SQLite synchronous pragmas, direct-to-storage data streams, and pre-mutation state rollbacks—you can achieve higher reliability with zero external dependencies than complex, multi-container tracking servers provide.

### What Comes Next in the Roadmap:
1. **Phase 5 — UI Component Library & Web Dashboard (`apps/web` & `@runvault/ui`):** A high-performance, offline-capable dashboard built with Vite, React, and Tailwind CSS to visualize local experiment curves and checkpoint lineages without requiring a remote server.
2. **Phase 6 — Hosted Control Plane:** Cloud storage integrations for AWS S3 and Google Cloud Storage, with optional multi-tenant team synchronization.

---

## Getting Started

RunVault is published on PyPI. You can install it today with:

```bash
# Base lightweight core (stdlib only)
pip install runvault-ai

# With PyTorch and Google Drive cloud integrations
pip install "runvault-ai[torch,gdrive]"
```

Drop it into your PyTorch training loop in 4 lines of code:

```python
import runvault
from runvault.integrations.torch import TorchCheckpointManager, restore_torch_checkpoint

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

with runvault.init("cifar10-resnet", name="baseline-run") as run:
    # Automatically restores newest valid checkpoint if previously interrupted
    latest = run.get_latest_valid_checkpoint()
    if latest:
        restore_torch_checkpoint(run, model, optimizer=optimizer)

    manager = TorchCheckpointManager(run=run, model=model, optimizer=optimizer, step_interval=500)
    for epoch in range(1, 11):
        for step, (inputs, targets) in enumerate(dataloader):
            # ... training step ...
            manager.step_end(global_step, metrics={"train_loss": loss.item()})
            global_step += 1
```

Explore the full open-source codebase, architecture decision records, and test suite on GitHub:
👉 [**https://github.com/Black-fox17/run-vault**](https://github.com/Black-fox17/run-vault)
