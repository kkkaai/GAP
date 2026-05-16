from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from prosthetic_grasp.common.artifacts import save_pipeline_artifacts
from prosthetic_grasp.common.io import save_image
from prosthetic_grasp.common.types import SensorFrame
from prosthetic_grasp.config.settings import load_settings
from prosthetic_grasp.pipeline import ProstheticGraspPipeline


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream Intel RealSense RGB frames. Press Enter to capture one frame and run the pipeline; press Esc to exit."
    )
    parser.add_argument("--config", default="config/default.toml", help="Path to the TOML config file.")
    parser.add_argument("--output-dir", default="outputs/realsense", help="Directory for captured artifacts.")
    parser.add_argument("--width", type=int, default=640, help="RGB stream width.")
    parser.add_argument("--height", type=int, default=480, help="RGB stream height.")
    parser.add_argument("--fps", type=int, default=30, help="RGB stream FPS.")
    return parser


def _make_capture_dir(base_output_dir: str | Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(base_output_dir) / timestamp


def main() -> None:
    args = build_argparser().parse_args()
    settings = load_settings(args.config)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the RealSense viewer. Install opencv-python first.") from exc

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required for the RealSense camera app. Install Intel RealSense SDK Python bindings."
        ) from exc

    phase1_mode = settings.get("phase1_mask", {}).get("mode", "unknown")
    print(f"Loaded phase1 mode: {phase1_mode}")
    if phase1_mode == "precomputed":
        print("Warning: phase1 is still configured for a precomputed mask. Captured frames will reach phase1,")
        print("but phase1 itself will not segment the live image until you switch to a real mask backend.")

    pipeline = ProstheticGraspPipeline(settings)

    pipeline_rs = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline_rs.start(config_rs)

    window_name = "GAP RealSense Capture"
    print("Press Enter to capture current frame and run the pipeline. Press Esc to exit.")

    try:
        while True:
            frames = pipeline_rs.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            cv2.imshow(window_name, color_bgr)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key in (10, 13):
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                capture_dir = _make_capture_dir(args.output_dir)
                capture_dir.mkdir(parents=True, exist_ok=True)
                rgb_path = capture_dir / "input_rgb.png"
                save_image(rgb_path, color_rgb)

                frame = SensorFrame(
                    rgb=color_rgb,
                    depth=None,
                    timestamp=time.time(),
                    rgb_path=str(rgb_path),
                    depth_path=None,
                )
                print(f"Captured frame -> {capture_dir}")
                result = pipeline.run(frame=frame)
                artifact_dir = save_pipeline_artifacts(result, capture_dir)
                print(f"Pipeline status: {result.status}")
                print(f"Artifacts saved to: {artifact_dir}")

    finally:
        pipeline_rs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
