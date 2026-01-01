import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class Wan22DistillConfig:
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    width: int = 1280
    height: int = 720
    num_frames: int = 81
    sample_shift: float = 5.0
    boundary_step_index: int = 2
    denoising_step_list: tuple[int, int, int, int] = (1000, 750, 500, 250)


class Wan22LightX2VGenerator:
    """LightX2V wrapper for Wan2.2 T2V 4-step distillation with Dual-LoRA switching."""

    PROMPT_PREFIX = "Cinematic shot, 4k, high motion, "

    def __init__(
        self,
        models_dir: Union[str, Path] = "./models",
        base_model_dirname: str = "Wan2.2-T2V-A14B",
        high_noise_lora_filename: str = "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        low_noise_lora_filename: str = "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors",
        offload_model: bool = False,  # Changed default to False to keep on GPU
        config: Wan22DistillConfig = Wan22DistillConfig(),
        attn_mode: str = "flash_attn2",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Wan2.2 generation")

        if config.num_inference_steps != 4:
            raise ValueError("Wan2.2 4-step distilled LoRAs require num_inference_steps=4")

        if float(config.guidance_scale) != 1.0:
            raise ValueError("Wan2.2 distilled models require guidance_scale=1.0")

        os.environ.setdefault("DTYPE", "BF16")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.models_dir = Path(models_dir)
        self.base_model_path = self.models_dir / base_model_dirname
        self.high_noise_lora_path = self.models_dir / "loras" / high_noise_lora_filename
        self.low_noise_lora_path = self.models_dir / "loras" / low_noise_lora_filename

        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Base model not found at {self.base_model_path}")
        if not self.high_noise_lora_path.exists():
            raise FileNotFoundError(f"High-noise LoRA not found at {self.high_noise_lora_path}")
        if not self.low_noise_lora_path.exists():
            raise FileNotFoundError(f"Low-noise LoRA not found at {self.low_noise_lora_path}")

        try:
            from lightx2v import LightX2VPipeline
        except Exception as e:
            raise RuntimeError(
                "Failed to import LightX2V "
                f"({type(e).__name__}: {e}). "
                "This usually means LightX2V wasn't installed successfully in the active environment, "
                "or a build/runtime dependency is missing. "
                "Re-run setup_wan.sh (it installs LightX2V into .venv) or install manually with: "
                "pip install -v git+https://github.com/ModelTC/LightX2V.git"
            ) from e

        self.pipe = LightX2VPipeline(
            task="t2v",
            model_path=str(self.base_model_path),
            model_cls="wan2.2_moe_distill",
        )

        if offload_model:
            self.pipe.enable_offload(
                cpu_offload=True,
                offload_granularity="model",
                text_encoder_offload=False,
                image_encoder_offload=False,
                vae_offload=False,
            )

        # LightX2V's LoRA API and expected LoRA names have changed across versions.
        # For Wan2.2 distill runners, LightX2V requires explicit LoRA names.
        # In particular, its Wan2.2 model code indexes lora_config["name"] directly.
        # If we pass dicts without a "name" key (e.g. {"lora_name": ...}), it can crash
        # later with KeyError("name") during model initialization.
        high_path = str(self.high_noise_lora_path)
        low_path = str(self.low_noise_lora_path)

        # Use the official expected names.
        # See LightX2V's example: examples/wan/wan_i2v_with_distill_loras.py
        self.pipe.enable_lora(
            [
                {"name": "high_noise_model", "path": high_path, "strength": 1.0},
                {"name": "low_noise_model", "path": low_path, "strength": 1.0},
            ]
        )

        if attn_mode == "flash_attn2":
            try:
                import flash_attn  # noqa: F401
            except Exception:
                attn_mode = "sdpa"

        self.pipe.create_generator(
            attn_mode=attn_mode,
            infer_steps=config.num_inference_steps,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            guidance_scale=config.guidance_scale,
            sample_shift=config.sample_shift,
            boundary_step_index=config.boundary_step_index,
            denoising_step_list=list(config.denoising_step_list),
        )

        self.config = config

    def generate_clip(self, prompt: str, output_path: Union[str, Path], seed: Optional[int] = None) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        import imageio
        import numpy as np
        from collections.abc import Sequence
        import shutil
        import subprocess
        from contextlib import contextmanager

        try:
            from PIL import Image as pil_image_module  # type: ignore
        except Exception:  # pragma: no cover
            pil_image_module = None  # type: ignore

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        enhanced_prompt = f"{self.PROMPT_PREFIX}{prompt.strip()}"
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        start = time.time()

        @contextmanager
        def _force_imageio_mp4_quality():
            """Best-effort hijack if the backend uses imageio to write MP4s.

            If LightX2V calls imageio.mimsave internally, this forces high-quality
            H.264 settings (CRF 18) at the FIRST encode (before any quality loss).
            If LightX2V doesn't use imageio for saving, this has no effect.
            """

            state = {"called": False}
            original = getattr(imageio, "mimsave", None)
            if original is None:
                yield state
                return

            def _strip_ffmpeg_params(params: list[str], flags_with_values: set[str]) -> list[str]:
                out: list[str] = []
                skip_next = False
                for p in params:
                    if skip_next:
                        skip_next = False
                        continue
                    if p in flags_with_values:
                        skip_next = True
                        continue
                    out.append(p)
                return out

            def patched(uri, ims, *args, **kwargs):
                state["called"] = True

                uri_s = str(uri)
                if uri_s.lower().endswith(".mp4"):
                    kwargs.setdefault("plugin", "ffmpeg")

                    forced = [
                        "-crf",
                        "18",
                        "-preset",
                        "slow",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                    ]

                    existing = kwargs.get("ffmpeg_params")
                    if existing is None:
                        kwargs["ffmpeg_params"] = forced
                    else:
                        existing_list = list(existing)
                        existing_list = _strip_ffmpeg_params(
                            existing_list,
                            flags_with_values={"-crf", "-preset", "-c:v", "-pix_fmt"},
                        )
                        kwargs["ffmpeg_params"] = existing_list + forced

                return original(uri, ims, *args, **kwargs)

            imageio.mimsave = patched  # type: ignore[assignment]
            try:
                yield state
            finally:
                imageio.mimsave = original  # type: ignore[assignment]
        
        def _reencode_mp4(input_path: Path, output_path: Path) -> None:
            tmp_out = output_path.with_suffix(output_path.suffix + ".reencode.tmp")
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_out),
            ]
            subprocess.run(cmd, check=True)
            tmp_out.replace(output_path)

        def _find_newest_mp4(search_dir: Path, since_ts: float) -> Path | None:
            candidates: list[Path] = []
            if not search_dir.exists():
                return None
            for p in search_dir.glob("*.mp4"):
                try:
                    if p.stat().st_mtime >= since_ts - 1.0:
                        candidates.append(p)
                except OSError:
                    continue
            if not candidates:
                return None
            return max(candidates, key=lambda p: p.stat().st_mtime)

        # 1. Generate frames (LightX2V return type can vary by version)
        raw = self.pipe.generate(
            seed=seed,
            prompt=enhanced_prompt,
            negative_prompt="",
            save_result_path=None, 
        )

        def _to_uint8_rgb_ndarray(frame: object) -> np.ndarray:
            """Normalize a single frame to uint8 HxWx3 RGB ndarray."""
            if isinstance(frame, torch.Tensor):
                t = frame.detach().cpu()
                if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                    # C,H,W -> H,W,C
                    t = t.permute(1, 2, 0)
                arr = t.numpy()
            elif isinstance(frame, np.ndarray):
                arr = frame
            elif pil_image_module is not None and isinstance(frame, pil_image_module.Image):
                arr = np.asarray(frame.convert("RGB"))
            else:
                # Last-resort conversion (may still fail for unknown types)
                arr = np.asarray(frame)

            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)

            if arr.ndim != 3:
                raise TypeError(f"Frame must be 3D (H,W,C); got shape {getattr(arr, 'shape', None)}")

            # Channel handling
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[-1] != 3:
                raise TypeError(f"Frame must have 1/3/4 channels; got {arr.shape[-1]}")

            # Dtype/range handling
            if arr.dtype != np.uint8:
                arr_f = arr.astype(np.float32, copy=False)
                max_val = float(np.nanmax(arr_f)) if arr_f.size else 0.0
                # Heuristic: if it's 0..1 floats, scale to 0..255
                if max_val <= 1.5:
                    arr_f = arr_f * 255.0
                arr_f = np.clip(arr_f, 0.0, 255.0)
                arr = arr_f.astype(np.uint8)

            return arr

        def _extract_frames(obj: object) -> list[np.ndarray]:
            """Extract list of frames from LightX2V output across versions."""
            if obj is None:
                raise RuntimeError(
                    "LightX2V returned None for frames. "
                    "This build may only support saving via save_result_path."
                )

            # Some versions may return (frames, meta) or similar
            if isinstance(obj, tuple) and obj:
                obj = obj[0]

            # Some versions may wrap frames in a dict
            if isinstance(obj, dict):
                for key in ("frames", "images", "video", "result"):
                    if key in obj:
                        obj = obj[key]
                        break

            # If it returns a saved path
            if isinstance(obj, (str, Path)):
                p = Path(obj)
                if p.exists():
                    # Backend wrote a file; copy it into the requested output path.
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    if p.resolve() != output_path.resolve():
                        shutil.copy2(p, output_path)
                    return []
                raise RuntimeError(f"LightX2V returned a path that does not exist: {p}")

            # Torch tensors: could be [F,C,H,W], [F,H,W,C], or [B,F,C,H,W]
            if isinstance(obj, torch.Tensor):
                t = obj.detach().cpu()
                if t.ndim == 5:
                    t = t[0]
                if t.ndim != 4:
                    raise TypeError(f"Unexpected tensor shape for video frames: {tuple(t.shape)}")

                # If [F,C,H,W] -> [F,H,W,C]
                if t.shape[1] in (1, 3, 4):
                    t = t.permute(0, 2, 3, 1)
                arr = t.numpy()

                # Split into list of frames
                return [_to_uint8_rgb_ndarray(arr[i]) for i in range(arr.shape[0])]

            # Numpy array: could be [F,H,W,C] or [F,C,H,W]
            if isinstance(obj, np.ndarray):
                arr = obj
                if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
                    return [_to_uint8_rgb_ndarray(arr[i]) for i in range(arr.shape[0])]
                if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
                    # [F,C,H,W] -> iterate and permute per frame
                    frames: list[np.ndarray] = []
                    for i in range(arr.shape[0]):
                        frames.append(_to_uint8_rgb_ndarray(np.transpose(arr[i], (1, 2, 0))))
                    return frames

            # Sequence of frames
            if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray, str)):
                frames_list = list(obj)
                if not frames_list:
                    raise RuntimeError("LightX2V returned an empty frame sequence")
                return [_to_uint8_rgb_ndarray(f) for f in frames_list]

            raise TypeError(f"Unsupported LightX2V frame output type: {type(obj).__name__}")

        # Some LightX2V builds return None and only support saving to disk.
        if raw is None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            since = time.time()

            # Attempt 1: some versions accept a full file path.
            patch_state = {"called": False}
            try:
                with _force_imageio_mp4_quality() as patch_state:
                    _ = self.pipe.generate(
                        seed=seed,
                        prompt=enhanced_prompt,
                        negative_prompt="",
                        save_result_path=str(output_path),
                    )
            except Exception:
                pass

            if output_path.exists():
                # If we successfully hijacked imageio inside the backend, avoid double-encoding.
                if not patch_state.get("called"):
                    _reencode_mp4(output_path, output_path)
                video_frames = []
            else:
                # Attempt 2: pass a directory and discover the created mp4.
                with _force_imageio_mp4_quality() as patch_state:
                    _ = self.pipe.generate(
                        seed=seed,
                        prompt=enhanced_prompt,
                        negative_prompt="",
                        save_result_path=str(output_path.parent),
                    )
                saved = _find_newest_mp4(output_path.parent, since_ts=since)
                if saved is None:
                    raise RuntimeError(
                        "LightX2V returned None for frames and did not write an MP4 to disk. "
                        "Try updating LightX2V or check its output directory configuration."
                    )
                if saved.resolve() != output_path.resolve():
                    shutil.copy2(saved, output_path)
                if not patch_state.get("called"):
                    _reencode_mp4(output_path, output_path)
                video_frames = []
        else:
            video_frames = _extract_frames(raw)

        # 3. Manually save with High Quality settings
        # If backend already wrote a file (rare), skip re-encoding.
        if video_frames:
            imageio.mimsave(
                str(output_path),
                video_frames,
                fps=24,
                plugin="ffmpeg",
                ffmpeg_params=[
                    "-crf",
                    "18",
                    "-preset",
                    "slow",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                ],
            )
        
        _ = time.time() - start

        if not output_path.exists():
            raise RuntimeError(f"Generation completed but output not found: {output_path}")
        if output_path.stat().st_size < 10_000:
            raise RuntimeError(f"Generated file too small ({output_path.stat().st_size} bytes): {output_path}")

        return str(output_path)