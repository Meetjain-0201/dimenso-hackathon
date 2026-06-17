# Live architecture — recon of Motion-Copilot + proposal for dimenso

**Recon + feasibility only. No code built; the dimenso pipeline is unchanged.**
Goal: make our offline engine (MediaPipe → IK → render, sequential, can't hold
30 fps) run *live*, reusing the real-time **pattern** from Motion-Copilot's live
"mimic" (NOT its IK — theirs is POV-camera + browser, different math).

---

## 1. Motion-Copilot's live path — files + how it hits real time

Motion-Copilot has **two** distinct paths; only the first is the relevant pattern:

### A. Live "mimic" (the real-time pattern we want) — **fully client-side, no server in the loop**
| File | Role |
|---|---|
| `src/web/react-app/src/components/buzz/scene/pose/MediaPipeRunner.ts` | webcam → MediaPipe `PoseLandmarker`+`HandLandmarker` (WASM/GPU), rAF detect loop |
| `…/buzz/scene/MirrorController.ts` | orchestrator: consumes detections, retargets, smooths, holds latest joints |
| `…/buzz/scene/pose/smoothing.ts` | **One Euro filter** (Casiez et al. 2012), one per joint |
| `…/buzz/scene/pose/handRetarget.ts`, `retarget.ts` | landmark → joint angles (their IK — **ignored** per brief) |
| `…/buzz/scene/URDFRig.ts` | applies joint values to the in-browser Three.js robot rig |

**How it stays real time** (verified in source):
- **Two decoupled loops + a latest-value buffer (NOT a queue).**
  - *Perception loop* (`MediaPipeRunner.tick`, `requestAnimationFrame`): each tick calls `detectForVideo()` on **the newest video frame only**, guarded by a strictly-monotonic `performance.now()` timestamp (`if (ts <= lastTimestamp) return`). It never queues — when inference is slow the next rAF simply fires later, so **late frames are implicitly dropped** and it always works on the freshest frame. Per-frame inference errors are caught and skipped, never kill the loop.
  - *Render loop* (Three.js rAF via `MirrorController.tick`): every display frame it reads the **latest** `this.joints` and applies them to the rig. It **never blocks on perception**; if no fresh detection it holds, and on subject-lost it lerps to neutral (`k=0.06`).
- **One Euro smoothing on timestamped samples** bridges the variable perception cadence to the smooth 60 fps render (smooth at rest, snappy on motion).
- Net: perception writes the latest joint target whenever it can; the robot loop consumes the latest value at display rate. **Producer/consumer fully decoupled by a single overwriting buffer.**

### B. Server-streamed sim (`cloud/g1_mujoco_server.py`, ~7700 lines) — NOT the mimic, but documents the transport gotcha
- Backend = **asyncio `websockets` server** (`asyncio.run(server.run())`). Robot runs headless (`MUJOCO_GL=egl`), frames are `cv2.imencode(".jpg")` (SIMD C path, releases GIL) and pushed over the WebSocket (`stream_frames` task). Heavy synchronous work is offloaded with `run_in_executor`.
- **Key gotcha, in their own words (line ~1276):** *"MuJoCo's EGL context is bound to the thread that created it, so creating a new Renderer in another thread races with the stream loop's renderer."* → **a MuJoCo GL context is single-thread-owned.** Only one thread may render.

### Reusable skeleton (plain terms)
> Run perception and the robot as **two independent loops** sharing **one overwriting "latest target" buffer**. Perception always processes the **newest** frame and drops the rest (drive off a monotonic clock); the robot loop reads the latest target every tick and never waits. Put a **One Euro filter** between them. Keep **all MuJoCo rendering on a single thread** (its GL context is thread-bound).

---

## 2. Proposed dimenso live architecture

Our setup differs from both Motion-Copilot paths:
- Perception is **Python** (MediaPipe + our DLS IK), not browser JS.
- The robot is the **native MuJoCo window** (`launch_passive` on `scene_g1_inspire.xml`), **not** browser-rendered and **not** streamed.
- The web panel only shows the **annotated source video** (CPU/`cv2` overlay) — **no MuJoCo frames cross the wire**, so we need only **one** GL context (the viewer) and the transport is trivial.

### Threads / processes / transport
```
                         ┌──────────────────────── shared state (locked, latest-value) ─────────────────┐
                         │   latest_qpos: (nq,) + ts      latest_jpeg: bytes      ctrl: {playing, paths} │
                         └───────▲───────────────────────────▲─────────────────────────▲────────────────┘
                                 │ write                      │ write                    │ read/write
   ┌─────────────────────────────┴──────┐      ┌──────────────┴───────────┐   ┌──────────┴───────────────┐
   │ SOLVER THREAD (or process)         │      │ MAIN THREAD               │   │ WEB-SERVER THREAD         │
   │ video clock → due frame (DROP late)│      │ mujoco.viewer.launch_     │   │ uvicorn/FastAPI           │
   │ MediaPipe Hands → stabilize → IK   │      │   passive(model,data)     │   │  • serves panel (upload   │
   │   (its OWN MjModel/MjData)          │      │ loop: read latest_qpos →  │   │    base.mp4+imu.json,Play)│
   │ write latest_qpos                   │      │   data.qpos (legs frozen) │   │  • POST /play → ctrl flag │
   │ cv2-annotate frame → latest_jpeg    │      │   → mj_forward → sync()   │   │  • GET /stream → MJPEG of │
   │ NO GL / NO render here              │      │ ONLY thread touching GL   │   │    latest_jpeg (multipart)│
   └─────────────────────────────────────┘      └───────────────────────────┘   └───────────────────────────┘
```
- **Main thread** = the passive viewer + sim/sync loop. Reads the latest joint-target buffer, applies (legs pinned to the stand keyframe), `mj_forward`, `sync()`, sleeps to ~60 fps. Holds / interpolates the last target if none is fresh — **never blocks on the solver**. Owns the **only** GL context.
- **Solver thread** = our existing per-frame chain (`pose_extract → stabilize → imu_sync → retarget`) wrapped in a loop, using a **separate** `MjModel`/`MjData` for IK (never the viewer's). Writes `latest_qpos`; also writes the annotated JPEG. Does **no** rendering.
- **Web server** = FastAPI on **uvicorn in a background thread** (must set `server.install_signal_handlers = lambda: None`, since signal handlers only register on the main thread). Transport to the browser = **MJPEG `multipart/x-mixed-replace`** (or a WebSocket) of `latest_jpeg`; the Play button flips a shared flag the solver reads.
- **Real-time / frame-drop policy:** solver computes the due video time from wall-clock-since-Play and **skips to that frame, dropping any it passed** — the robot stays synced to real time instead of lagging. (Mirrors Motion-Copilot's "newest frame only" rule, applied to a file clock instead of a webcam.)

### Launch sequence (one command)
`python -m sim.live_engine <recording>`:
1. start uvicorn FastAPI in a daemon thread (panel + `/play` + `/stream`);
2. start the solver thread (idle until Play);
3. call `mujoco.viewer.launch_passive(...)` on the **main thread** and enter the sim/sync loop (blocks main thread, as required).

Open the printed panel URL; the MuJoCo window is already up. Upload → Play → annotated video plays in the panel while the native window shows the G1+Inspire moving live. Meet sets the MuJoCo camera to POV by hand in the window.

### Main-thread / GL conflict — resolution
- `launch_passive` keeps the **main thread** and its **single** GL context (on Linux/glfw this is fine; macOS would additionally need `mjpython` — not our case).
- uvicorn runs in a **non-main thread** with signal handlers disabled — a standard, supported pattern.
- The solver **never renders** (only `cv2` annotation, pure CPU), so **no second GL context is ever created** — which is exactly the race Motion-Copilot warns about. This is the clean resolution and is only possible *because* Meet chose not to stream the MuJoCo view.
- The solver's IK uses its **own** `MjData`, so there is **no cross-thread MuJoCo state**; the viewer only consumes a plain `qpos` array through the lock.

### Reuse vs build-new
| Reuse (pattern or existing code) | Build new |
|---|---|
| Decoupled loops + single latest-value buffer (MC mimic) | Threaded harness: main viewer loop + solver thread + uvicorn thread + locked buffers |
| Drop-to-newest-frame scheduling (MC), retimed to a file clock | FastAPI panel: upload, Play, MJPEG `/stream` of annotated video |
| One Euro filter — **already in our `pose_extract.py`** | Wall-clock→video-frame scheduler with frame drop |
| Our `retarget.py` (DLS IK, stow, rate-limit) + `replay_g1.py` viewer loop | Split solve vs view into threads; second `MjData` for IK |

**Differs from Motion-Copilot:** they render the robot in-browser (mimic) or stream MuJoCo frames over WebSocket (server path); we use the **native window** and stream only the **annotated video** (one-way MJPEG). Their perception is browser-JS; ours is Python (so the GIL matters — see risks).

---

## 3. Risks / unknowns
- **GIL / CPU contention (biggest):** MediaPipe (~0.1 s/frame) + DLS IK in a Python *thread* share the GIL with the viewer + uvicorn. MediaPipe and `cv2` release the GIL in C, and numpy partly does, so it likely sustains a usable rate — but if the solver can't keep ~real time, frame-drop keeps the robot *synced* while the *update rate* (motion smoothness/fidelity) degrades. **Mitigation:** run the solver as a separate **process** (`multiprocessing` + shared-memory `qpos`), fully sidestepping GIL contention. Recommended if a single process stutters.
- **launch_passive must own the main thread** — constrains structure (server + solver must be the background threads), but is satisfiable on Linux. Not a blocker.
- **uvicorn-in-a-thread** needs `install_signal_handlers=lambda:None`; Ctrl-C then routes only through the viewer window.
- **Wall-clock vs video clock drift / seeking cost:** `cv2` random-seek on mp4 can be slow on some codecs; sequential read + skip-decode is safer than per-frame `set(POS_FRAMES)`.
- **IMU in live mode:** our offline IMU sync resamples the whole stream onto frame times; live needs a streaming/windowed variant (minor adaptation, not architectural).
- **First-frame warm-up / camera framing:** the native window opens before Play; Meet positions the POV camera manually (acceptable by design).

## VERDICT
**Live feasible: YES.** The decoupled latest-value pattern makes the native MuJoCo window smooth regardless of solver speed, and streaming only the annotated video means a single GL context and a trivial transport — all reusing Motion-Copilot's proven skeleton plus code we already have.
**Single biggest risk:** Python **GIL/CPU contention** between MediaPipe+IK and the viewer/server in one process; if it stutters, move the solver to a separate process with shared-memory joint targets.
