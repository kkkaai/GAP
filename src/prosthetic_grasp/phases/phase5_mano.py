# from __future__ import annotations

# import sys
# from contextlib import contextmanager
# from dataclasses import dataclass
# from os import chdir, getcwd
# from pathlib import Path
# from typing import Any

# import numpy as np

# from prosthetic_grasp.common.types import Phase5HandPrediction, Phase5ManoResult


# @contextmanager
# def _pushd(path: Path):
#     old_cwd = getcwd()
#     chdir(path)
#     try:
#         yield
#     finally:
#         chdir(old_cwd)


# @dataclass
# class Phase5ManoConfig:
#     hamer_root: str = "external/hamer"
#     checkpoint: str | None = None
#     body_detector: str = "vitdet"
#     batch_size: int = 8
#     rescale_factor: float = 2.0
#     detector_score_threshold: float = 0.5
#     hand_keypoint_score_threshold: float = 0.5
#     min_hand_keypoints: int = 4
#     hand_side: str = "right"
#     download_models: bool = True

#     def __post_init__(self) -> None:
#         self.hamer_root = self.hamer_root.strip()
#         if self.checkpoint is not None:
#             self.checkpoint = self.checkpoint.strip() or None
#         self.body_detector = self.body_detector.strip().lower()
#         self.hand_side = self.hand_side.strip().lower()
#         if self.body_detector not in {"vitdet", "regnety"}:
#             raise ValueError(f"body_detector must be 'vitdet' or 'regnety', got {self.body_detector!r}.")
#         if self.batch_size <= 0:
#             raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
#         if self.rescale_factor <= 0:
#             raise ValueError(f"rescale_factor must be positive, got {self.rescale_factor}.")
#         if not 0.0 <= self.detector_score_threshold <= 1.0:
#             raise ValueError(f"detector_score_threshold must be in [0, 1], got {self.detector_score_threshold}.")
#         if not 0.0 <= self.hand_keypoint_score_threshold <= 1.0:
#             raise ValueError(
#                 f"hand_keypoint_score_threshold must be in [0, 1], got {self.hand_keypoint_score_threshold}."
#             )
#         if self.min_hand_keypoints <= 0:
#             raise ValueError(f"min_hand_keypoints must be positive, got {self.min_hand_keypoints}.")
#         if self.hand_side not in {"right", "left", "both"}:
#             raise ValueError(f"hand_side must be 'right', 'left', or 'both', got {self.hand_side!r}.")


# class Phase5Mano:
#     """Recover MANO hand parameters and mesh from a generated 2D hand image using HaMeR."""

#     def __init__(self, config: Phase5ManoConfig | None = None) -> None:
#         self.config = config or Phase5ManoConfig()
#         self._loaded = False
#         self._device = None
#         self._model = None
#         self._model_cfg = None
#         self._detector = None
#         self._pose_model = None
#         self._faces = None
#         self._cam_crop_to_full = None
#         self._recursive_to = None
#         self._vitdet_dataset_cls = None
#         self._vitpose_model_cls = None
#         self._hamer_root: Path | None = None

#     def run(self, image_rgb: np.ndarray) -> Phase5ManoResult:
#         self._validate_image(image_rgb)
#         self._ensure_loaded()

#         img_cv2 = np.ascontiguousarray(image_rgb[:, :, ::-1])
#         detections = self._detect_hand_boxes(img_cv2, image_rgb)
#         if not detections:
#             return Phase5ManoResult(
#                 status="no_detections",
#                 message="HaMeR did not find any hand boxes.",
#                 faces=self._faces,
#                 hands=[],
#             )

#         boxes = np.stack([det["bbox"] for det in detections]).astype(np.float32)
#         right = np.asarray([1 if det["is_right"] else 0 for det in detections], dtype=np.float32)
#         dataset = self._vitdet_dataset_cls(
#             self._model_cfg,
#             img_cv2,
#             boxes,
#             right,
#             rescale_factor=self.config.rescale_factor,
#         )

#         import torch

#         dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=self.config.batch_size,
#             shuffle=False,
#             num_workers=0,
#         )

#         hands: list[Phase5HandPrediction] = []
#         for batch in dataloader:
#             batch = self._recursive_to(batch, self._device)
#             with torch.no_grad():
#                 out = self._model(batch)

#             pred_cam = out["pred_cam"].clone()
#             multiplier = 2 * batch["right"] - 1
#             pred_cam[:, 1] = multiplier * pred_cam[:, 1]
#             box_center = batch["box_center"].float()
#             box_size = batch["box_size"].float()
#             img_size = batch["img_size"].float()
#             scaled_focal_length = (
#                 self._model_cfg.EXTRA.FOCAL_LENGTH / self._model_cfg.MODEL.IMAGE_SIZE * img_size.max()
#             )
#             pred_cam_t_full = self._cam_crop_to_full(
#                 pred_cam,
#                 box_center,
#                 box_size,
#                 img_size,
#                 scaled_focal_length,
#             )

#             batch_size = batch["img"].shape[0]
#             for n in range(batch_size):
#                 hand_index = int(batch["personid"][n].detach().cpu().item())
#                 detection = detections[hand_index]
#                 is_right = bool(batch["right"][n].detach().cpu().item())

#                 vertices = out["pred_vertices"][n].detach().cpu().numpy()
#                 vertices[:, 0] = (2 * float(is_right) - 1) * vertices[:, 0]

#                 hands.append(
#                     Phase5HandPrediction(
#                         hand_index=hand_index,
#                         is_right=is_right,
#                         bbox_xyxy=detection["bbox"].astype(np.float32),
#                         keypoints_2d=detection["keypoints_2d"].astype(np.float32),
#                         keypoint_score_mean=float(detection["score_mean"]),
#                         vertices=vertices,
#                         keypoints_3d=out["pred_keypoints_3d"][n].detach().cpu().numpy(),
#                         pred_cam=pred_cam[n].detach().cpu().numpy(),
#                         pred_cam_t_crop=out["pred_cam_t"][n].detach().cpu().numpy(),
#                         pred_cam_t_full=pred_cam_t_full[n].detach().cpu().numpy(),
#                         focal_length=float(scaled_focal_length.detach().cpu().item()),
#                         mano_params=self._extract_mano_params(out["pred_mano_params"], n),
#                     )
#                 )

#         return Phase5ManoResult(
#             status="ok",
#             message=f"Recovered MANO predictions for {len(hands)} hand(s).",
#             faces=self._faces,
#             hands=hands,
#         )

#     @staticmethod
#     def _validate_image(image_rgb: np.ndarray) -> None:
#         if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
#             raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_rgb.shape}.")

#     def _ensure_loaded(self) -> None:
#         if self._loaded:
#             return

#         hamer_root = Path(self.config.hamer_root).expanduser().resolve()
#         if not hamer_root.exists():
#             raise FileNotFoundError(f"HaMeR root does not exist: {hamer_root}")
#         hamer_root_str = str(hamer_root)
#         if hamer_root_str not in sys.path:
#             sys.path.insert(0, hamer_root_str)

#         import hamer
#         import torch
#         from hamer.configs import CACHE_DIR_HAMER
#         from hamer.datasets.vitdet_dataset import ViTDetDataset
#         from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
#         from hamer.utils import recursive_to
#         from hamer.utils.renderer import cam_crop_to_full
#         from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
#         from vitpose_model import ViTPoseModel

#         with _pushd(hamer_root):
#             if self.config.download_models:
#                 download_models(CACHE_DIR_HAMER)

#             checkpoint = self.config.checkpoint or DEFAULT_CHECKPOINT
#             model, model_cfg = load_hamer(checkpoint)
#         device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#         model = model.to(device)
#         model.eval()

#         if self.config.body_detector == "vitdet":
#             from detectron2.config import LazyConfig

#             cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
#             detectron2_cfg = LazyConfig.load(str(cfg_path))
#             detectron2_cfg.train.init_checkpoint = (
#                 "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
#                 "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
#             )
#             for i in range(3):
#                 detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = self.config.detector_score_threshold
#             detector = DefaultPredictor_Lazy(detectron2_cfg)
#         else:
#             from detectron2 import model_zoo

#             detectron2_cfg = model_zoo.get_config(
#                 "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
#                 trained=True,
#             )
#             detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = self.config.detector_score_threshold
#             detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
#             detector = DefaultPredictor_Lazy(detectron2_cfg)

#         self._device = device
#         self._model = model
#         self._model_cfg = model_cfg
#         self._detector = detector
#         self._pose_model = None
#         self._vitpose_model_cls = ViTPoseModel
#         self._hamer_root = hamer_root
#         self._faces = np.asarray(model.mano.faces, dtype=np.int32)
#         self._cam_crop_to_full = cam_crop_to_full
#         self._recursive_to = recursive_to
#         self._vitdet_dataset_cls = ViTDetDataset
#         self._loaded = True

#     def _ensure_pose_model(self):
#         if self._pose_model is not None:
#             return self._pose_model
#         if self._vitpose_model_cls is None or self._hamer_root is None:
#             raise RuntimeError("HaMeR must be loaded before ViTPose can be initialized.")
#         with _pushd(self._hamer_root):
#             self._pose_model = self._vitpose_model_cls(self._device)
#         return self._pose_model

#     def _detect_hand_boxes(self, img_cv2: np.ndarray, image_rgb: np.ndarray) -> list[dict[str, Any]]:
#         det_out = self._detector(img_cv2)
#         det_instances = det_out["instances"]
#         valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > self.config.detector_score_threshold)
#         pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].detach().cpu().numpy()
#         pred_scores = det_instances.scores[valid_idx].detach().cpu().numpy()
#         if len(pred_bboxes) == 0:
#             return []

#         pose_model = self._ensure_pose_model()
#         vitposes_out = pose_model.predict_pose(
#             image_rgb,
#             [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
#         )

#         detections: list[dict[str, Any]] = []
#         for vitposes in vitposes_out:
#             if self.config.hand_side in {"left", "both"}:
#                 self._append_hand_detection(detections, vitposes["keypoints"][-42:-21], is_right=False)
#             if self.config.hand_side in {"right", "both"}:
#                 self._append_hand_detection(detections, vitposes["keypoints"][-21:], is_right=True)
#         return detections

#     def _append_hand_detection(self, detections: list[dict[str, Any]], keypoints: np.ndarray, is_right: bool) -> None:
#         valid = keypoints[:, 2] > self.config.hand_keypoint_score_threshold
#         if int(valid.sum()) < self.config.min_hand_keypoints:
#             return
#         bbox = np.array(
#             [
#                 keypoints[valid, 0].min(),
#                 keypoints[valid, 1].min(),
#                 keypoints[valid, 0].max(),
#                 keypoints[valid, 1].max(),
#             ],
#             dtype=np.float32,
#         )
#         detections.append(
#             {
#                 "bbox": bbox,
#                 "is_right": is_right,
#                 "keypoints_2d": keypoints,
#                 "score_mean": float(keypoints[valid, 2].mean()),
#             }
#         )

#     @staticmethod
#     def _extract_mano_params(pred_mano_params: dict[str, Any], index: int) -> dict[str, np.ndarray]:
#         return {
#             name: value[index].detach().cpu().numpy()
#             for name, value in pred_mano_params.items()
#         }


from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from os import chdir, getcwd
from pathlib import Path
from typing import Any

import numpy as np

from prosthetic_grasp.common.types import Phase5HandPrediction, Phase5ManoResult

@contextmanager
def _pushd(path: Path):
    old_cwd = getcwd()
    chdir(path)
    try:
        yield
    finally:
        chdir(old_cwd)

@dataclass
class Phase5ManoConfig:
    hamer_root: str = "external/hamer"
    checkpoint: str | None = None
    body_detector: str = "vitdet"
    batch_size: int = 8
    rescale_factor: float = 2.0
    detector_score_threshold: float = 0.5
    hand_keypoint_score_threshold: float = 0.5
    min_hand_keypoints: int = 4
    hand_side: str = "right"
    download_models: bool = True

    def __post_init__(self) -> None:
        self.hamer_root = self.hamer_root.strip()
        if self.checkpoint is not None:
            self.checkpoint = self.checkpoint.strip() or None
        self.body_detector = self.body_detector.strip().lower()
        self.hand_side = self.hand_side.strip().lower()
        if self.body_detector not in {"vitdet", "regnety"}:
            raise ValueError(f"body_detector must be 'vitdet' or 'regnety', got {self.body_detector!r}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.rescale_factor <= 0:
            raise ValueError(f"rescale_factor must be positive, got {self.rescale_factor}.")
        if not 0.0 <= self.detector_score_threshold <= 1.0:
            raise ValueError(f"detector_score_threshold must be in [0, 1], got {self.detector_score_threshold}.")
        if not 0.0 <= self.hand_keypoint_score_threshold <= 1.0:
            raise ValueError(
                f"hand_keypoint_score_threshold must be in [0, 1], got {self.hand_keypoint_score_threshold}."
            )
        if self.min_hand_keypoints <= 0:
            raise ValueError(f"min_hand_keypoints must be positive, got {self.min_hand_keypoints}.")
        if self.hand_side not in {"right", "left", "both"}:
            raise ValueError(f"hand_side must be 'right', 'left', or 'both', got {self.hand_side!r}.")

class Phase5Mano:
    """Recover MANO hand parameters and mesh from a generated 2D hand image using HaMeR."""

    def __init__(self, config: Phase5ManoConfig | None = None) -> None:
        self.config = config or Phase5ManoConfig()
        self._loaded = False
        self._device = None
        self._model = None
        self._model_cfg = None
        self._detector = None
        self._pose_model = None
        self._faces = None
        self._cam_crop_to_full = None
        self._recursive_to = None
        self._vitdet_dataset_cls = None

    def run(self, image_rgb: np.ndarray) -> Phase5ManoResult:
        self._validate_image(image_rgb)
        self._ensure_loaded()

        img_cv2 = np.ascontiguousarray(image_rgb[:, :, ::-1])
        detections = self._detect_hand_boxes(img_cv2, image_rgb)
        if not detections:
            return Phase5ManoResult(
                status="no_detections",
                message="HaMeR did not find any hand boxes.",
                faces=self._faces,
                hands=[],
            )

        boxes = np.stack([det["bbox"] for det in detections]).astype(np.float32)
        right = np.asarray([1 if det["is_right"] else 0 for det in detections], dtype=np.float32)
        dataset = self._vitdet_dataset_cls(
            self._model_cfg,
            img_cv2,
            boxes,
            right,
            rescale_factor=self.config.rescale_factor,
        )

        import torch

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        hands: list[Phase5HandPrediction] = []
        for batch in dataloader:
            batch = self._recursive_to(batch, self._device)
            with torch.no_grad():
                out = self._model(batch)

            pred_cam = out["pred_cam"].clone()
            multiplier = 2 * batch["right"] - 1
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self._model_cfg.EXTRA.FOCAL_LENGTH / self._model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            pred_cam_t_full = self._cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            )

            batch_size = batch["img"].shape[0]
            for n in range(batch_size):
                hand_index = int(batch["personid"][n].detach().cpu().item())
                detection = detections[hand_index]
                is_right = bool(batch["right"][n].detach().cpu().item())

                vertices = out["pred_vertices"][n].detach().cpu().numpy()
                vertices[:, 0] = (2 * float(is_right) - 1) * vertices[:, 0]

                hands.append(
                    Phase5HandPrediction(
                        hand_index=hand_index,
                        is_right=is_right,
                        bbox_xyxy=detection["bbox"].astype(np.float32),
                        keypoints_2d=detection["keypoints_2d"].astype(np.float32),
                        keypoint_score_mean=float(detection["score_mean"]),
                        vertices=vertices,
                        keypoints_3d=out["pred_keypoints_3d"][n].detach().cpu().numpy(),
                        pred_cam=pred_cam[n].detach().cpu().numpy(),
                        pred_cam_t_crop=out["pred_cam_t"][n].detach().cpu().numpy(),
                        pred_cam_t_full=pred_cam_t_full[n].detach().cpu().numpy(),
                        focal_length=float(scaled_focal_length.detach().cpu().item()),
                        mano_params=self._extract_mano_params(out["pred_mano_params"], n),
                    )
                )

        return Phase5ManoResult(
            status="ok",
            message=f"Recovered MANO predictions for {len(hands)} hand(s).",
            faces=self._faces,
            hands=hands,
        )

    @staticmethod
    def _validate_image(image_rgb: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image_rgb.shape}.")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        hamer_root = Path(self.config.hamer_root).expanduser().resolve()
        if not hamer_root.exists():
            raise FileNotFoundError(f"HaMeR root does not exist: {hamer_root}")
        hamer_root_str = str(hamer_root)
        if hamer_root_str not in sys.path:
            sys.path.insert(0, hamer_root_str)

        import hamer
        import torch
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.models import download_models, load_hamer
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel

        if self.config.download_models:
            from hamer.configs import CACHE_DIR_HAMER

            with _pushd(hamer_root):
                download_models(CACHE_DIR_HAMER)

        checkpoint = self.config.checkpoint
        if checkpoint is None:
            checkpoint = hamer_root / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
        else:
            checkpoint = Path(checkpoint).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = hamer_root / checkpoint
        checkpoint = str(checkpoint.resolve())
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                "HaMeR checkpoint not found. Set phase5_mano.checkpoint or place the checkpoint at "
                f"{hamer_root / '_DATA' / 'hamer_ckpts' / 'checkpoints' / 'hamer.ckpt'}."
            )
        with _pushd(hamer_root):
            model, model_cfg = load_hamer(checkpoint)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        model.eval()

        if self.config.body_detector == "vitdet":
            from detectron2.config import LazyConfig

            cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = (
                "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            )
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = self.config.detector_score_threshold
            detector = DefaultPredictor_Lazy(detectron2_cfg)
        else:
            from detectron2 import model_zoo

            detectron2_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
                trained=True,
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = self.config.detector_score_threshold
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            detector = DefaultPredictor_Lazy(detectron2_cfg)

        self._device = device
        self._model = model
        self._model_cfg = model_cfg
        self._detector = detector
        with _pushd(hamer_root):
            self._pose_model = ViTPoseModel(device)
        self._faces = np.asarray(model.mano.faces, dtype=np.int32)
        self._cam_crop_to_full = cam_crop_to_full
        self._recursive_to = recursive_to
        self._vitdet_dataset_cls = ViTDetDataset
        self._loaded = True

    def _detect_hand_boxes(self, img_cv2: np.ndarray, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        det_out = self._detector(img_cv2)
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > self.config.detector_score_threshold)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].detach().cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].detach().cpu().numpy()
        if len(pred_bboxes) == 0:
            return []

        vitposes_out = self._pose_model.predict_pose(
            image_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        detections: list[dict[str, Any]] = []
        for vitposes in vitposes_out:
            if self.config.hand_side in {"left", "both"}:
                self._append_hand_detection(detections, vitposes["keypoints"][-42:-21], is_right=False)
            if self.config.hand_side in {"right", "both"}:
                self._append_hand_detection(detections, vitposes["keypoints"][-21:], is_right=True)
        return detections

    def _append_hand_detection(self, detections: list[dict[str, Any]], keypoints: np.ndarray, is_right: bool) -> None:
        valid = keypoints[:, 2] > self.config.hand_keypoint_score_threshold
        if int(valid.sum()) < self.config.min_hand_keypoints:
            return
        bbox = np.array(
            [
                keypoints[valid, 0].min(),
                keypoints[valid, 1].min(),
                keypoints[valid, 0].max(),
                keypoints[valid, 1].max(),
            ],
            dtype=np.float32,
        )
        detections.append(
            {
                "bbox": bbox,
                "is_right": is_right,
                "keypoints_2d": keypoints,
                "score_mean": float(keypoints[valid, 2].mean()),
            }
        )

    @staticmethod
    def _extract_mano_params(pred_mano_params: dict[str, Any], index: int) -> dict[str, np.ndarray]:
        return {
            name: value[index].detach().cpu().numpy()
            for name, value in pred_mano_params.items()
        }
