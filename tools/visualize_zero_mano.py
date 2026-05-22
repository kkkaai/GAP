from pathlib import Path

import torch
import smplx
import trimesh


MANO_MODEL_DIR = Path("models/mano")  # 改成你的 MANO 模型文件目录
HAND_SIDE = "right"


def main():
    model = smplx.create(
        model_path=str(MANO_MODEL_DIR),
        model_type="mano",
        is_rhand=(HAND_SIDE == "right"),
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    )

    betas = torch.zeros(1, 10)
    global_orient = torch.zeros(1, 3)
    hand_pose = torch.zeros(1, 45)
    transl = torch.zeros(1, 3)

    output = model(
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl,
        return_verts=True,
        return_full_pose=True,
    )

    vertices = output.vertices.detach().cpu().numpy()[0]
    joints = output.joints.detach().cpu().numpy()[0]
    faces = model.faces

    print("vertices:", vertices.shape)
    print("joints:", joints.shape)
    print("faces:", faces.shape)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    # 加一些小球标出 joints
    for i, p in enumerate(joints):
        sphere = trimesh.creation.icosphere(radius=0.003)
        sphere.apply_translation(p)
        scene.add_geometry(sphere, node_name=f"joint_{i}")

    scene.show()


if __name__ == "__main__":
    main()