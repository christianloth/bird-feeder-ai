"""
Bird species classifier supporting multiple backends.

Runs a fine-tuned model (MobileNetV2 or EfficientNet-B2) on cropped bird images. Supports:
- PyTorch (MPS/CUDA/CPU) — for Mac development and GPU servers
- ONNX Runtime — cross-platform, optimized inference
- Hailo NPU — Raspberry Pi AI HAT+ production deployment
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.training.transforms import get_inference_transforms

logger = logging.getLogger(__name__)


class BirdClassifier:
    """
    Species classifier with pluggable backends.

    Usage:
        classifier = BirdClassifier.from_pytorch("models/bird-classifier/mobilenetv2/2026-03-26_12-43/best_model.pth")
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
        confidence_threshold: float = 0.10,
        input_size: int = 224,
    ):
        self.backend = backend
        self._model = model
        self._ort_session = ort_session
        self._hef_model = hef_model
        self.class_names = class_names or {}
        self.device = device
        self.confidence_threshold = confidence_threshold
        self._transform = get_inference_transforms(input_size)

    @classmethod
    def from_pytorch(
        cls,
        checkpoint_path: str | Path,
        class_names: dict[int, str] | None = None,
        num_classes: int = 555,
        device: str | None = None,
        confidence_threshold: float = 0.10,
        model_name: str = "efficientnet_b2",
    ) -> "BirdClassifier":
        """Load a PyTorch checkpoint. Auto-selects MPS/CUDA/CPU."""
        from src.training.model import create_model, get_model_config

        model_config = get_model_config(model_name)

        if device is None:
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)

        model = create_model(
            num_classes=num_classes, pretrained=False,
            freeze_backbone=False, model_name=model_name,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=dev, weights_only=True))
        model = model.to(dev)
        model.eval()

        logger.info(f"Loaded {model_name} classifier on {dev} from {checkpoint_path}")
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
        confidence_threshold: float = 0.10,
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
        confidence_threshold: float = 0.10,
    ) -> "BirdClassifier":
        """Load a Hailo HEF model for NPU inference on Raspberry Pi."""
        from hailo_platform import HEF, VDevice, ConfigureParams

        hef = HEF(str(hef_path))
        vdevice = VDevice()
        infer_model = vdevice.configure(hef, ConfigureParams.create_from_hef(hef))

        logger.info(f"Loaded Hailo classifier from {hef_path}")
        return cls(
            backend="hailo",
            hef_model={"hef": hef, "vdevice": vdevice, "infer_model": infer_model},
            class_names=class_names,
            confidence_threshold=confidence_threshold,
        )

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        """Convert a BGR numpy crop to a normalized tensor."""
        # BGR → RGB → PIL
        crop_rgb = crop_bgr[:, :, ::-1]
        pil_image = Image.fromarray(crop_rgb)
        # Apply inference transforms (Resize→CenterCrop→ToTensor→Normalize)
        tensor = self._transform(pil_image)
        return tensor

    def _predict_pytorch(self, tensor: torch.Tensor) -> tuple[int, float]:
        """Run inference with PyTorch backend."""
        batch = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self._model(batch)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        return predicted.item(), confidence.item()

    def _predict_onnx(self, tensor: torch.Tensor) -> tuple[int, float]:
        """Run inference with ONNX Runtime backend."""
        input_name = self._ort_session.get_inputs()[0].name
        input_np = tensor.unsqueeze(0).numpy()
        outputs = self._ort_session.run(None, {input_name: input_np})
        logits = outputs[0][0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])
        return predicted, confidence

    def _predict_hailo(self, tensor: torch.Tensor) -> tuple[int, float]:
        """Run inference with Hailo NPU backend."""
        from hailo_platform import InferVStreams

        input_np = tensor.unsqueeze(0).numpy()
        infer_model = self._hef_model["infer_model"]

        with InferVStreams(infer_model) as pipeline:
            input_data = {infer_model.input_vstream_infos[0].name: input_np}
            results = pipeline.infer(input_data)
            output_name = infer_model.output_vstream_infos[0].name
            logits = results[output_name][0]

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])
        return predicted, confidence

    def predict(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        """
        Classify a bird crop image.

        Args:
            crop_bgr: BGR numpy array from OpenCV (the cropped bird region).

        Returns:
            (species_name, confidence) or (None, 0.0) if below threshold.
        """
        tensor = self._preprocess_crop(crop_bgr)

        if self.backend == "pytorch":
            class_idx, confidence = self._predict_pytorch(tensor)
        elif self.backend == "onnx":
            class_idx, confidence = self._predict_onnx(tensor)
        elif self.backend == "hailo":
            class_idx, confidence = self._predict_hailo(tensor)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        species_name = self.class_names.get(class_idx, f"class_{class_idx}")
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
