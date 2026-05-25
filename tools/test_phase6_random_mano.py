from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import smplx
import torch

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from prosthetic_grasp.common.types import Phase5HandPrediction, Phase5ManoResult
from prosthetic_grasp.phases.phase6_prosthetic_action import (
    Phase6ProstheticAction,
    Phase6ProstheticActionConfig,
    _build_hand_frame,
)


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
MANO_TIP_VERTEX_IDS = np.array([744, 320, 443, 554, 671], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test src Phase6 retargeting with a random synthetic MANO hand."
    )
    parser.add_argument("--mano-root", default="models", help="Root containing mano/MANO_RIGHT.pkl.")
    parser.add_argument("--robot-profile", default="shadow_hand")
    parser.add_argument("--output-dir", default="outputs/phase6_random_mano")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pose-scale", type=float, default=0.45)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=250)
    parser.add_argument("--regularization-weight", type=float, default=0.002)
    return parser.parse_args()


def make_mano_21_joints(vertices: np.ndarray, joints16: np.ndarray) -> np.ndarray:
    tips = vertices[MANO_TIP_VERTEX_IDS]
    joints21 = np.empty((21, 3), dtype=np.float64)
    joints21[0] = joints16[0]
    joints21[1:4] = joints16[13:16]
    joints21[4] = tips[0]
    joints21[5:8] = joints16[1:4]
    joints21[8] = tips[1]
    joints21[9:12] = joints16[4:7]
    joints21[12] = tips[2]
    joints21[13:16] = joints16[10:13]
    joints21[16] = tips[3]
    joints21[17:20] = joints16[7:10]
    joints21[20] = tips[4]
    return joints21


def create_random_phase5_result(mano_root: str, seed: int, pose_scale: float) -> tuple[Phase5ManoResult, np.ndarray]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = smplx.create(
        model_path=mano_root,
        model_type="mano",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    )

    hand_pose = torch.randn(1, 45, generator=generator) * pose_scale
    hand_pose[:, 2::3] = hand_pose[:, 2::3].abs()
    output = model(
        betas=torch.zeros(1, 10),
        global_orient=torch.zeros(1, 3),
        hand_pose=hand_pose,
        transl=torch.zeros(1, 3),
        return_verts=True,
        return_full_pose=True,
    )

    vertices = output.vertices.detach().cpu().numpy()[0].astype(np.float64)
    joints16 = output.joints.detach().cpu().numpy()[0].astype(np.float64)
    joints21 = make_mano_21_joints(vertices, joints16)

    hand = Phase5HandPrediction(
        hand_index=0,
        is_right=True,
        bbox_xyxy=np.zeros(4, dtype=np.float32),
        keypoints_2d=np.zeros((21, 2), dtype=np.float32),
        keypoint_score_mean=1.0,
        vertices=vertices.astype(np.float32),
        keypoints_3d=joints21.astype(np.float32),
        pred_cam=np.zeros(3, dtype=np.float32),
        pred_cam_t_crop=np.zeros(3, dtype=np.float32),
        pred_cam_t_full=np.zeros(3, dtype=np.float32),
        focal_length=0.0,
        mano_params={
            "hand_pose": hand_pose.detach().cpu().numpy()[0].astype(np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "global_orient": np.zeros(3, dtype=np.float32),
        },
    )
    phase5 = Phase5ManoResult(
        status="ok",
        message="Synthetic random MANO hand for Phase6 testing.",
        faces=np.asarray(model.faces, dtype=np.int64),
        hands=[hand],
    )
    return phase5, vertices


def robot_mesh(phase6: Phase6ProstheticAction, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = phase6._ensure_model()
    if hasattr(model, "mesh_vertices"):
        return model.mesh_vertices(np.asarray(action, dtype=np.float64))
    model.robot.update_cfg(dict(zip(model.joint_names, np.asarray(action, dtype=np.float64))))
    world_to_wrist = np.linalg.inv(model.robot.get_transform(model.wrist_link))
    mesh = model.robot.scene.to_geometry()
    vertices_world = np.asarray(mesh.vertices, dtype=np.float64)
    vertices_h = np.concatenate([vertices_world, np.ones((vertices_world.shape[0], 1))], axis=1)
    vertices_wrist = (world_to_wrist @ vertices_h.T).T[:, :3]
    return vertices_wrist, np.asarray(mesh.faces, dtype=np.int64)


def map_mano_points_like_phase6(
    phase6: Phase6ProstheticAction,
    hand: Phase5HandPrediction,
    points: np.ndarray,
) -> np.ndarray:
    model = phase6._ensure_model()
    robot_open = model.forward(model.zero_action)["fingertips"]
    keypoints = np.asarray(hand.keypoints_3d, dtype=np.float64)
    if not hand.is_right:
        keypoints = keypoints.copy()
        keypoints[:, 0] *= -1.0

    mano_frame = _build_hand_frame(
        wrist=keypoints[0],
        index_mcp=keypoints[5],
        middle_mcp=keypoints[9],
        little_mcp=keypoints[17],
    )
    robot_frame = _build_hand_frame(
        wrist=np.zeros(3),
        index_mcp=robot_open[1],
        middle_mcp=robot_open[2],
        little_mcp=robot_open[4],
    )
    mano_fingertips = keypoints[[4, 8, 12, 16, 20]]
    mano_tip_rel = mano_fingertips - keypoints[0]
    mano_mid = max(float(np.linalg.norm(mano_tip_rel[2])), 1e-8)
    robot_mid = max(float(np.linalg.norm(robot_open[2])), 1e-8)
    scale = float(np.clip(robot_mid / mano_mid, phase6.config.min_scale, phase6.config.max_scale))
    point_rel = np.asarray(points, dtype=np.float64) - keypoints[0]
    return ((point_rel @ mano_frame) * scale) @ robot_frame.T


def write_visualization(
    output_path: Path,
    mano_vertices_robot: np.ndarray,
    mano_faces: np.ndarray,
    robot_vertices: np.ndarray,
    robot_faces: np.ndarray,
    target_tips: np.ndarray,
    robot_tips: np.ndarray,
    mano_wrist: np.ndarray,
    robot_wrist: np.ndarray,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=mano_vertices_robot[:, 0],
            y=mano_vertices_robot[:, 1],
            z=mano_vertices_robot[:, 2],
            i=mano_faces[:, 0],
            j=mano_faces[:, 1],
            k=mano_faces[:, 2],
            name="MANO mesh mapped to robot wrist frame",
            color="rgba(40, 180, 80, 0.28)",
            opacity=0.28,
        )
    )
    fig.add_trace(
        go.Mesh3d(
            x=robot_vertices[:, 0],
            y=robot_vertices[:, 1],
            z=robot_vertices[:, 2],
            i=robot_faces[:, 0],
            j=robot_faces[:, 1],
            k=robot_faces[:, 2],
            name="Robot mesh optimized by src Phase6",
            color="rgba(40, 100, 230, 0.42)",
            opacity=0.42,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=target_tips[:, 0],
            y=target_tips[:, 1],
            z=target_tips[:, 2],
            mode="markers+text",
            name="Phase6 target MANO fingertips",
            text=FINGER_NAMES,
            marker=dict(size=7, color="red"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[robot_wrist[0]],
            y=[robot_wrist[1]],
            z=[robot_wrist[2]],
            mode="markers+text",
            name="Robot wrist link origin",
            text=["★ robot base_link"],
            marker=dict(size=11, color="blue", symbol="diamond"),
            textfont=dict(size=16, color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[mano_wrist[0]],
            y=[mano_wrist[1]],
            z=[mano_wrist[2]],
            mode="markers+text",
            name="MANO wrist joint",
            text=["★ MANO wrist"],
            marker=dict(size=11, color="red", symbol="diamond"),
            textfont=dict(size=16, color="red"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=robot_tips[:, 0],
            y=robot_tips[:, 1],
            z=robot_tips[:, 2],
            mode="markers+text",
            name="Phase6 robot fingertips",
            text=FINGER_NAMES,
            marker=dict(size=7, color="blue"),
        )
    )
    for index, name in enumerate(FINGER_NAMES):
        pair = np.stack([target_tips[index], robot_tips[index]], axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=pair[:, 0],
                y=pair[:, 1],
                z=pair[:, 2],
                mode="lines",
                name=f"{name} error",
                line=dict(color="rgba(30,30,30,0.45)", width=4),
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Random MANO retargeted by src Phase6",
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase5_result, mano_vertices = create_random_phase5_result(args.mano_root, args.seed, args.pose_scale)
    config = Phase6ProstheticActionConfig(
        robot_profile=args.robot_profile,
        tip_point="mesh_tip",
        optimization_restarts=args.restarts,
        max_nfev=args.max_nfev,
        regularization_weight=args.regularization_weight,
        random_seed=args.seed,
    )
    phase6 = Phase6ProstheticAction(config)
    result = phase6.run(phase5_result)
    if result.status != "ok":
        raise RuntimeError(f"Phase6 failed: {result.status}: {result.message}")

    hand = phase5_result.hands[0]
    mano_vertices_robot = map_mano_points_like_phase6(phase6, hand, mano_vertices)
    mano_joints_robot = map_mano_points_like_phase6(phase6, hand, hand.keypoints_3d)
    robot_vertices, robot_faces = robot_mesh(phase6, result.action)
    output_html = output_dir / f"random_mano_{args.robot_profile}_phase6_overlay.html"
    write_visualization(
        output_html,
        mano_vertices_robot,
        np.asarray(phase5_result.faces, dtype=np.int64),
        robot_vertices,
        robot_faces,
        np.asarray(result.target_fingertips_wrist, dtype=np.float64),
        np.asarray(result.prosthetic_fingertips_wrist, dtype=np.float64),
        mano_joints_robot[0],
        np.zeros(3, dtype=np.float64),
    )

    summary = {
        "source": "src/prosthetic_grasp/phases/phase6_prosthetic_action.py",
        "robot_profile": args.robot_profile,
        "seed": args.seed,
        "pose_scale": args.pose_scale,
        "status": result.status,
        "message": result.message,
        "finger_names": FINGER_NAMES,
        "action_names": result.action_names,
        "action": result.action.astype(float).tolist(),
        "error_m": result.fingertip_error.astype(float).tolist(),
        "mean_error_m": float(np.mean(result.fingertip_error)),
        "max_error_m": float(np.max(result.fingertip_error)),
        "metadata": result.metadata,
        "output_html": str(output_html),
    }
    summary_path = output_dir / f"random_mano_{args.robot_profile}_phase6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Retarget source: src/prosthetic_grasp/phases/phase6_prosthetic_action.py")
    print(f"Robot profile: {args.robot_profile}")
    for name, err in zip(FINGER_NAMES, result.fingertip_error):
        print(f"{name:>6}: {err * 1000.0:8.3f} mm")
    print(f"  mean: {np.mean(result.fingertip_error) * 1000.0:8.3f} mm")
    print(f"   max: {np.max(result.fingertip_error) * 1000.0:8.3f} mm")
    print(f"summary: {summary_path}")
    print(f"visualization: {output_html}")


if __name__ == "__main__":
    main()
