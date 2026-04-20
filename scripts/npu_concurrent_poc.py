"""
POC: concurrent Hailo-10H NPU inference across two Python processes.

Demonstrates whether the shared VDevice (group_id="SHARED") mechanism in
HailoRT 5.x actually lets two separate processes hit the same physical NPU
at the same time. Parent spawns two child processes; each child loads a
different HEF and runs inference in a tight loop. We then compare:

  Phase 1 — YOLO alone:   baseline FPS for yolov11x
  Phase 2 — ViT alone:    baseline FPS for vit_base_birds
  Phase 3 — Both in parallel (two processes, shared VDevice):
            report each child's FPS; expect total throughput to
            ~= baseline (NPU is time-sliced at 40 TOPS, not parallel)

If Phase 3 produces real FPS for both children, cross-process shared
VDevice works on your setup. If the second child errors or hangs,
it doesn't (fallback: single-process multi-model).

USAGE:
    # NPU must be free — stop the pipeline first:
    pkill -f "src.pipeline.pipeline"
    # Then:
    .venv/bin/python scripts/npu_concurrent_poc.py
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np


HEF_DIR = Path(__file__).parent.parent / "models" / "hef"
YOLO_HEF = HEF_DIR / "yolov11x.hef"
VIT_HEF = HEF_DIR / "vit_base_birds.hef"


def run_inference_loop(
    hef_path: str,
    name: str,
    seconds: float,
    shared: bool,
    ready_event,
    start_event,
    result_queue,
):
    """Run inference in a tight loop for `seconds`, report FPS.

    All boilerplate (VDevice, InferModel, bindings, buffers) is set up,
    then the loop waits for `start_event` so all workers kick off together.
    """
    from hailo_platform import VDevice, FormatType, HailoSchedulingAlgorithm

    params = VDevice.create_params()
    if shared:
        # The key flag for cross-process NPU sharing on Hailo-10H.
        # Both child processes pass the same group_id so HailoRT's runtime
        # binds them to the same underlying VDevice + scheduler.
        params.group_id = "SHARED"
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    try:
        vdevice = VDevice(params)
    except Exception as e:
        result_queue.put({"name": name, "error": f"VDevice creation failed: {e!r}"})
        ready_event.set()
        return

    try:
        infer_model = vdevice.create_infer_model(hef_path)
        infer_model.input().set_format_type(FormatType.UINT8)
        for output in infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
        configured = infer_model.configure()

        # Pre-allocate a single binding with random input — the goal here is
        # to measure raw NPU throughput, not preprocessing.
        input_shape = infer_model.input().shape  # (H, W, C)
        input_buf = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)

        bindings = configured.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(input_buf))
        for out_name in infer_model.output_names:
            out_shape = infer_model.output(out_name).shape
            bindings.output(out_name).set_buffer(np.empty(out_shape, dtype=np.float32))

        # Warm-up: run a few inferences before timing
        for _ in range(3):
            configured.run([bindings], timeout=5000)

        ready_event.set()
        start_event.wait()  # synchronize start across workers

        count = 0
        t0 = time.perf_counter()
        t_end = t0 + seconds
        while time.perf_counter() < t_end:
            configured.run([bindings], timeout=5000)
            count += 1
        elapsed = time.perf_counter() - t0
        fps = count / elapsed
        result_queue.put({
            "name": name,
            "inferences": count,
            "elapsed_s": elapsed,
            "fps": fps,
        })
    except Exception as e:
        result_queue.put({"name": name, "error": f"inference failed: {e!r}"})
    finally:
        # Explicitly release — important on Hailo to avoid CIM lingering
        del vdevice


def phase(name: str, workers: list[dict], seconds: float, shared: bool):
    """Run one phase with given workers (possibly just one)."""
    print(f"\n=== {name} ===")
    ctx = mp.get_context("spawn")  # spawn avoids fork() issues with Hailo
    result_q = ctx.Queue()
    ready_events = [ctx.Event() for _ in workers]
    start_event = ctx.Event()

    procs = []
    for w, ready in zip(workers, ready_events):
        p = ctx.Process(
            target=run_inference_loop,
            args=(w["hef"], w["name"], seconds, shared, ready, start_event, result_q),
        )
        p.start()
        procs.append(p)

    # Wait for all workers to be warmed up
    for ready in ready_events:
        ready.wait(timeout=60)

    # Kick off the timed loops simultaneously
    start_event.set()

    for p in procs:
        p.join(timeout=seconds + 60)

    results = []
    while not result_q.empty():
        results.append(result_q.get())

    for r in results:
        if "error" in r:
            print(f"  [{r['name']}] ERROR: {r['error']}")
        else:
            print(
                f"  [{r['name']}] {r['inferences']} inferences "
                f"in {r['elapsed_s']:.2f}s = {r['fps']:.1f} FPS"
            )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="duration per phase")
    args = ap.parse_args()

    for p in (YOLO_HEF, VIT_HEF):
        if not p.exists():
            raise SystemExit(f"HEF not found: {p}")

    print(f"HEFs:")
    print(f"  YOLO: {YOLO_HEF}")
    print(f"  ViT:  {VIT_HEF}")
    print(f"Per-phase duration: {args.seconds}s")

    # Phase 1: YOLO alone (single process, UNIQUE group) — baseline
    yolo_alone = phase(
        "Phase 1: YOLO11x alone",
        workers=[{"name": "YOLO11x", "hef": str(YOLO_HEF)}],
        seconds=args.seconds,
        shared=False,
    )

    # Phase 2: ViT alone (single process, UNIQUE group) — baseline
    vit_alone = phase(
        "Phase 2: ViT-Base alone",
        workers=[{"name": "ViT-Base", "hef": str(VIT_HEF)}],
        seconds=args.seconds,
        shared=False,
    )

    # Phase 3: both concurrently, two processes, shared VDevice group
    both = phase(
        "Phase 3: YOLO + ViT concurrently (shared VDevice, 2 processes)",
        workers=[
            {"name": "YOLO11x", "hef": str(YOLO_HEF)},
            {"name": "ViT-Base", "hef": str(VIT_HEF)},
        ],
        seconds=args.seconds,
        shared=True,
    )

    # Summary
    print("\n=== Summary ===")

    def get_fps(results, name):
        for r in results:
            if r.get("name") == name and "fps" in r:
                return r["fps"]
        return None

    y_solo = get_fps(yolo_alone, "YOLO11x")
    v_solo = get_fps(vit_alone, "ViT-Base")
    y_conc = get_fps(both, "YOLO11x")
    v_conc = get_fps(both, "ViT-Base")

    if None in (y_solo, v_solo, y_conc, v_conc):
        print("  Some phases failed — shared VDevice may not be supported here.")
        return

    print(f"  YOLO solo:    {y_solo:.1f} FPS")
    print(f"  ViT  solo:    {v_solo:.1f} FPS")
    print(f"  YOLO concur:  {y_conc:.1f} FPS  ({y_conc/y_solo*100:.0f}% of solo)")
    print(f"  ViT  concur:  {v_conc:.1f} FPS  ({v_conc/v_solo*100:.0f}% of solo)")
    print(f"  Concurrent total throughput: "
          f"{y_conc + v_conc:.1f} inf/s "
          f"(YOLO solo alone = {y_solo:.1f})")

    # Interpretation
    if y_conc > 0.3 * y_solo and v_conc > 0.3 * v_solo:
        print("\n  ✓ Shared VDevice WORKS across processes. NPU time-slices "
              "between the two models via ROUND_ROBIN scheduling.")
    else:
        print("\n  ✗ Shared VDevice appears NOT to work reliably on this "
              "setup. Prefer single-process multi-model approach.")


if __name__ == "__main__":
    main()
