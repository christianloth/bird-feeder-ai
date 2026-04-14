"""
Bird species classifier supporting multiple backends.

Runs a fine-tuned model (ViT-Small or EfficientNet-Lite4) on cropped bird images. Supports:
- PyTorch (MPS/CUDA/CPU) — for Mac development and GPU servers
- ONNX Runtime — cross-platform, optimized inference
- Hailo NPU — Raspberry Pi AI HAT+ production deployment
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.training.transforms import get_inference_transforms

SLOW_CLASSIFY_MS = 200  # Log a warning if classification takes longer than this

logger = logging.getLogger(__name__)


class BirdClassifier:
    """
    Species classifier with pluggable backends.

    Usage:
        classifier = BirdClassifier.from_pytorch("models/bird-classifier/vit_small/best_model.pth")
        species, confidence = classifier.predict(bird_crop_bgr)
    """

    def __init__(
        self,
        backend: str,
        model: nn.Module | None = None,
        ort_session=None,
        hef_model=None,
        class_names: dict[int, str] | None = None,
        device: torch.device | None = None,
        confidence_threshold: float | None = None,
        input_size: int = 224,
    ):
        self.backend = backend
        self._model = model
        self._ort_session = ort_session
        self._hef_model = hef_model
        self.class_names = class_names or {}
        self.device = device
        from config.settings import settings
        self.confidence_threshold = confidence_threshold or settings.classification_confidence_threshold
        self._transform = get_inference_transforms(input_size)
        self._input_size = input_size

    @classmethod
    def from_pytorch(
        cls,
        checkpoint_path: str | Path,
        class_names: dict[int, str] | None = None,
        num_classes: int = 555,
        device: str | None = None,
        confidence_threshold: float | None = None,
        model_name: str = "vit_small",
    ) -> "BirdClassifier":
        """Load a PyTorch checkpoint. Auto-selects MPS/CUDA/CPU."""
        from src.training.model import create_model, get_model_config

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.critical(f"Classifier checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

        model_config = get_model_config(model_name)

        from config.settings import get_device

        dev = torch.device(get_device(device))

        logger.debug(f"Loading {model_name} classifier from {checkpoint_path}")
        model = create_model(
            num_classes=num_classes, pretrained=False,
            freeze_backbone=False, model_name=model_name,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev, weights_only=True))
        model = model.to(dev)
        model.eval()

        logger.info(
            f"Loaded {model_name} classifier on {dev} from {checkpoint_path} "
            f"({num_classes} classes, threshold={confidence_threshold})"
        )
        return cls(
            backend="pytorch",
            model=model,
            class_names=class_names,
            device=dev,
            confidence_threshold=confidence_threshold,
            input_size=model_config["input_size"],
        )

    @classmethod
    def from_onnx(
        cls,
        onnx_path: str | Path,
        class_names: dict[int, str] | None = None,
        confidence_threshold: float | None = None,
    ) -> "BirdClassifier":
        """Load an ONNX model for cross-platform inference."""
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(onnx_path), providers=providers)

        logger.info(f"Loaded ONNX classifier from {onnx_path}")
        return cls(
            backend="onnx",
            ort_session=session,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
        )

    @classmethod
    def from_hailo(
        cls,
        hef_path: str | Path,
        class_names: dict[int, str] | None = None,
        confidence_threshold: float | None = None,
        vdevice=None,
    ) -> "BirdClassifier":
        """Load a Hailo HEF model for NPU inference on Raspberry Pi."""
        from hailo_platform import VDevice, FormatType

        if not Path(hef_path).exists():
            logger.critical(f"Classifier HEF model not found: {hef_path}")
            raise FileNotFoundError(f"Classifier HEF model not found: {hef_path}")

        logger.debug(f"Loading Hailo classifier from {hef_path}")
        if vdevice is None:
            vdevice = VDevice()
        infer_model = vdevice.create_infer_model(str(hef_path))
        infer_model.input().set_format_type(FormatType.FLOAT32)
        infer_model.output().set_format_type(FormatType.FLOAT32)
        configured_model = infer_model.configure()

        input_shape = infer_model.input().shape  # (H, W, C)
        logger.info(
            f"Loaded Hailo classifier from {hef_path} "
            f"(input={input_shape}, output={infer_model.output().shape})"
        )
        return cls(
            backend="hailo",
            hef_model={
                "vdevice": vdevice,
                "infer_model": infer_model,
                "configured": configured_model,
            },
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            input_size=input_shape[0],
        )

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Convert a BGR numpy crop to a normalized tensor for PyTorch/ONNX backends."""
        # BGR → RGB → PIL
        crop_rgb = crop_bgr[:, :, ::-1]
        pil_image = Image.fromarray(crop_rgb)
        # Apply inference transforms (Resize→CenterCrop→ToTensor→Normalize)
        tensor = self._transform(pil_image)
        return tensor

    def _preprocess_crop_hailo(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR numpy crop to raw 0-255 pixels in HWC format for Hailo.

        The HEF has normalization baked in (the .alls model script applies
        (pixel - 127.5) / 127.5 on-chip). The host must send raw uint8 pixels
        with NO normalization — just direct resize to the input size.

        Uses direct Resize (not Resize+CenterCrop) because the input is already
        a tight YOLO crop — center-cropping would discard discriminative
        features at the edges (wing tips, tail, head). Matches val/inference
        transforms in transforms.py.
        """
        from torchvision import transforms

        crop_rgb = crop_bgr[:, :, ::-1]
        pil_image = Image.fromarray(crop_rgb)
        # Direct resize to square — no CenterCrop, no ToTensor, no Normalize
        spatial_transform = transforms.Resize((self._input_size, self._input_size))
        pil_resized = spatial_transform(pil_image)
        # Convert to float32 numpy in HWC format, values 0-255
        return np.array(pil_resized, dtype=np.float32)

    def _get_logits_pytorch(self, tensor: torch.Tensor) -> np.ndarray:
        """Get raw logits from PyTorch backend."""
        batch = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self._model(batch)
        return outputs[0].cpu().numpy()

    def _predict_pytorch(self, tensor: torch.Tensor) -> tuple[int, float]:
        """Run inference with PyTorch backend."""
        batch = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self._model(batch)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        return predicted.item(), confidence.item()

    def _get_logits_onnx(self, tensor: torch.Tensor) -> np.ndarray:
        """Get raw logits from ONNX backend."""
        input_name = self._ort_session.get_inputs()[0].name
        input_np = tensor.unsqueeze(0).numpy()
        outputs = self._ort_session.run(None, {input_name: input_np})
        return outputs[0][0]

    def _predict_onnx(self, tensor: torch.Tensor) -> tuple[int, float]:
        """Run inference with ONNX Runtime backend."""
        logits = self._get_logits_onnx(tensor)
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])
        return predicted, confidence

    def _get_logits_hailo(self, input_np: np.ndarray) -> np.ndarray:
        """Get raw logits from Hailo NPU backend (InferModel API).

        Args:
            input_np: Raw pixels in HWC format, float32, values 0-255.
                      NOT normalized — the HEF handles normalization on-chip.
        """
        infer_model = self._hef_model["infer_model"]
        configured = self._hef_model["configured"]

        input_np = np.ascontiguousarray(input_np, dtype=np.float32)

        bindings = configured.create_bindings()
        bindings.input().set_buffer(input_np)
        out_shape = infer_model.output().shape
        bindings.output().set_buffer(np.empty(out_shape, dtype=np.float32))
        configured.run([bindings], timeout=5000)

        return bindings.output().get_buffer().flatten()

    def _predict_hailo(self, input_np: np.ndarray) -> tuple[int, float]:
        """Run inference with Hailo NPU backend."""
        logits = self._get_logits_hailo(input_np)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])
        return predicted, confidence

    def predict_logits(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Return raw logits (before softmax) for a bird crop.

        Used for logit averaging across multiple frames — accumulate logits,
        average them, then apply softmax once for a more accurate prediction.

        Args:
            crop_bgr: BGR numpy array from OpenCV (the cropped bird region).

        Returns:
            1-D numpy array of logits (one per class).
        """
        if self.backend == "hailo":
            input_np = self._preprocess_crop_hailo(crop_bgr)
            return self._get_logits_hailo(input_np)

        tensor = self._preprocess_crop(crop_bgr)
        if self.backend == "pytorch":
            return self._get_logits_pytorch(tensor)
        elif self.backend == "onnx":
            return self._get_logits_onnx(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def predict(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        """
        Classify a bird crop image.

        Args:
            crop_bgr: BGR numpy array from OpenCV (the cropped bird region).

        Returns:
            (species_name, confidence) or (None, 0.0) if below threshold.
        """
        t0 = time.perf_counter()

        if self.backend == "hailo":
            input_np = self._preprocess_crop_hailo(crop_bgr)
            class_idx, confidence = self._predict_hailo(input_np)
        else:
            tensor = self._preprocess_crop(crop_bgr)
            if self.backend == "pytorch":
                class_idx, confidence = self._predict_pytorch(tensor)
            elif self.backend == "onnx":
                class_idx, confidence = self._predict_onnx(tensor)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        species_name = self.class_names.get(class_idx, f"class_{class_idx}")

        logger.debug(f"Classify: {species_name} ({confidence:.3f}) in {elapsed_ms:.1f}ms")
        if elapsed_ms > SLOW_CLASSIFY_MS:
            logger.warning(f"Slow classification: {elapsed_ms:.0f}ms (threshold={SLOW_CLASSIFY_MS}ms)")
        if confidence < self.confidence_threshold:
            logger.debug(
                f"Classification below threshold: {species_name} "
                f"({confidence:.3f} < {self.confidence_threshold})"
            )

        return species_name, confidence

    def predict_top_k(self, crop_bgr: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Return top-K predictions with species names and confidences."""
        tensor = self._preprocess_crop(crop_bgr)

        if self.backend == "pytorch":
            batch = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self._model(batch)
                probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = probs.topk(k)
            results = [
                (self.class_names.get(idx.item(), f"class_{idx.item()}"), prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
        else:
            # ONNX / Hailo fallback
            if self.backend == "onnx":
                input_name = self._ort_session.get_inputs()[0].name
                outputs = self._ort_session.run(None, {input_name: tensor.unsqueeze(0).numpy()})
                logits = outputs[0][0]
            else:
                raise NotImplementedError(f"top_k not yet implemented for {self.backend}")

            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            top_indices = np.argsort(probs)[-k:][::-1]
            results = [
                (self.class_names.get(int(i), f"class_{i}"), float(probs[i]))
                for i in top_indices
            ]

        return results
