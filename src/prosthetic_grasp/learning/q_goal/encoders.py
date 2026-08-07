from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def mlp(sizes: list[int], *, activation: type[nn.Module] = nn.SiLU, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@dataclass
class ConditionEncoderConfig:
    image_encoder: str = "small_cnn"
    image_feature_dim: int = 128
    image_embedding_dim: int = 0
    freeze_image_encoder: bool = True
    lollipop_mask_encoder: str = "small_cnn"
    lollipop_mask_feature_dim: int = 64
    use_lollipop_params: bool = True
    lollipop_param_dim: int = 6
    lollipop_param_feature_dim: int = 32
    task_encoder: str = "embedding"
    task_vocab_size: int = 1
    task_feature_dim: int = 128
    task_embedding_dim: int = 0
    siglip_model_id: str = "google/siglip-base-patch16-224"
    freeze_siglip: bool = True
    use_q_current: bool = True
    q_current_feature_dim: int = 64
    condition_dim: int = 512
    q_dim: int = 0
    dropout: float = 0.0


class SmallImageCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchvisionResNet18Encoder(nn.Module):
    def __init__(self, out_dim: int, *, freeze: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError("torchvision_resnet18 encoder requires torchvision.") from exc

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        in_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.project = nn.Linear(in_dim, out_dim)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = (rgb - self.mean) / self.std
        return self.project(self.backbone(rgb))


class SharedSigLIPEncoder(nn.Module):
    def __init__(self, model_id: str, *, freeze: bool = True) -> None:
        super().__init__()
        try:
            from transformers import AutoProcessor, SiglipModel
        except ImportError as exc:
            raise ImportError("SigLIP encoders require transformers.") from exc

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SiglipModel.from_pretrained(model_id)
        self.freeze = freeze
        image_processor = getattr(self.processor, "image_processor", None)
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        config = self.model.config
        text_config = getattr(config, "text_config", None)
        vision_config = getattr(config, "vision_config", None)
        self.out_dim = int(
            getattr(config, "projection_dim", 0)
            or getattr(text_config, "projection_size", 0)
            or getattr(vision_config, "projection_size", 0)
            or getattr(text_config, "hidden_size", 0)
            or getattr(vision_config, "hidden_size", 768)
        )

    def _maybe_no_grad(self):
        return torch.no_grad() if self.freeze else torch.enable_grad()

    @staticmethod
    def _as_feature_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        for attr in ("pooler_output", "image_embeds", "text_embeds"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value
        value = getattr(output, "last_hidden_state", None)
        if isinstance(value, torch.Tensor):
            return value.mean(dim=1)
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            value = output[0]
            return value.mean(dim=1) if value.ndim == 3 else value
        raise TypeError(f"Unsupported SigLIP output type: {type(output)!r}")

    def encode_image(self, rgb: torch.Tensor) -> torch.Tensor:
        pixel_values = (rgb - self.image_mean) / self.image_std
        with self._maybe_no_grad():
            return self._as_feature_tensor(self.model.get_image_features(pixel_values=pixel_values))

    def encode_text(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with self._maybe_no_grad():
            return self._as_feature_tensor(self.model.get_text_features(**encoded))


class ConditionEncoder(nn.Module):
    def __init__(self, config: ConditionEncoderConfig) -> None:
        super().__init__()
        self.config = config
        parts: list[int] = []
        self.siglip_encoder: SharedSigLIPEncoder | None = None
        if config.image_encoder == "siglip" or config.task_encoder == "siglip":
            self.siglip_encoder = SharedSigLIPEncoder(config.siglip_model_id, freeze=config.freeze_siglip)

        if config.image_encoder == "small_cnn":
            self.image_encoder = SmallImageCNN(3, config.image_feature_dim)
            parts.append(config.image_feature_dim)
        elif config.image_encoder == "torchvision_resnet18":
            self.image_encoder = TorchvisionResNet18Encoder(
                config.image_feature_dim,
                freeze=config.freeze_image_encoder,
            )
            parts.append(config.image_feature_dim)
        elif config.image_encoder == "precomputed":
            self.image_encoder = nn.Linear(config.image_embedding_dim, config.image_feature_dim)
            parts.append(config.image_feature_dim)
        elif config.image_encoder == "siglip":
            if self.siglip_encoder is None:
                raise RuntimeError("SigLIP encoder was not initialized.")
            self.image_encoder = nn.Linear(self.siglip_encoder.out_dim, config.image_feature_dim)
            parts.append(config.image_feature_dim)
        elif config.image_encoder == "none":
            self.image_encoder = None
        else:
            raise ValueError(f"Unknown image_encoder {config.image_encoder!r}.")

        if config.lollipop_mask_encoder == "small_cnn":
            self.mask_encoder = SmallImageCNN(1, config.lollipop_mask_feature_dim)
            parts.append(config.lollipop_mask_feature_dim)
        elif config.lollipop_mask_encoder == "none":
            self.mask_encoder = None
        else:
            raise ValueError(f"Unknown lollipop_mask_encoder {config.lollipop_mask_encoder!r}.")

        if config.use_lollipop_params:
            self.lollipop_param_encoder = mlp(
                [config.lollipop_param_dim, config.lollipop_param_feature_dim, config.lollipop_param_feature_dim],
                dropout=config.dropout,
            )
            parts.append(config.lollipop_param_feature_dim)
        else:
            self.lollipop_param_encoder = None

        if config.task_encoder == "embedding":
            self.task_encoder = nn.Embedding(max(config.task_vocab_size, 1), config.task_feature_dim)
            parts.append(config.task_feature_dim)
        elif config.task_encoder == "precomputed":
            self.task_encoder = nn.Linear(config.task_embedding_dim, config.task_feature_dim)
            parts.append(config.task_feature_dim)
        elif config.task_encoder == "siglip":
            if self.siglip_encoder is None:
                raise RuntimeError("SigLIP encoder was not initialized.")
            self.task_encoder = nn.Linear(self.siglip_encoder.out_dim, config.task_feature_dim)
            parts.append(config.task_feature_dim)
        elif config.task_encoder == "none":
            self.task_encoder = None
        else:
            raise ValueError(f"Unknown task_encoder {config.task_encoder!r}.")

        if config.use_q_current:
            if config.q_dim <= 0:
                raise ValueError("q_dim is required when use_q_current=True.")
            self.q_current_encoder = mlp([config.q_dim, config.q_current_feature_dim, config.q_current_feature_dim])
            parts.append(config.q_current_feature_dim)
        else:
            self.q_current_encoder = None

        if not parts:
            raise ValueError("At least one condition encoder branch must be enabled.")
        self.out_dim = config.condition_dim
        self.fusion = mlp([sum(parts), config.condition_dim, config.condition_dim], dropout=config.dropout)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        features: list[torch.Tensor] = []
        if self.image_encoder is not None:
            if self.config.image_encoder == "precomputed":
                features.append(self.image_encoder(batch["image_embedding"]))
            elif self.config.image_encoder == "siglip":
                if self.siglip_encoder is None:
                    raise RuntimeError("SigLIP encoder was not initialized.")
                features.append(self.image_encoder(self.siglip_encoder.encode_image(batch["rgb"])))
            else:
                features.append(self.image_encoder(batch["rgb"]))
        if self.mask_encoder is not None:
            features.append(self.mask_encoder(batch["lollipop_mask"]))
        if self.lollipop_param_encoder is not None:
            features.append(self.lollipop_param_encoder(batch["lollipop_params"]))
        if self.task_encoder is not None:
            if self.config.task_encoder == "precomputed":
                features.append(self.task_encoder(batch["task_embedding"]))
            elif self.config.task_encoder == "siglip":
                if self.siglip_encoder is None:
                    raise RuntimeError("SigLIP encoder was not initialized.")
                texts = batch["task_text"]
                if isinstance(texts, str):
                    texts = [texts]
                features.append(self.task_encoder(self.siglip_encoder.encode_text(list(texts), batch["rgb"].device)))
            else:
                features.append(self.task_encoder(batch["task_id"]).float())
        if self.q_current_encoder is not None:
            features.append(self.q_current_encoder(batch["q_current"]))
        return self.fusion(torch.cat(features, dim=-1))
