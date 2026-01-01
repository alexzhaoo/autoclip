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

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        enhanced_prompt = f"{self.PROMPT_PREFIX}{prompt.strip()}"
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        start = time.time()
        
        # 1. Generate frames (returns Tensors)
        video_frames = self.pipe.generate(
            seed=seed,
            prompt=enhanced_prompt,
            negative_prompt="",
            save_result_path=None, 
        )

        # 2. Conversion Logic: Handle Tensor -> Numpy List
        # If output is a single 4D Tensor [F, C, H, W] or [F, H, W, C]
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.detach().cpu()
            # If channels are first [F, C, H, W], permute to [F, H, W, C]
            if video_frames.ndim == 4 and video_frames.shape[1] == 3:
                 video_frames = video_frames.permute(0, 2, 3, 1)
            
            # Ensure 0-255 range and uint8 type
            if video_frames.dtype == torch.float32 or video_frames.dtype == torch.float16:
                video_frames = (video_frames * 255).clamp(0, 255).to(torch.uint8)
            
            video_frames = video_frames.numpy()
        
        # If output is a list of Tensors
        elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], torch.Tensor):
            processed_frames = []
            for frame in video_frames:
                frame = frame.detach().cpu()
                if frame.ndim == 3 and frame.shape[0] == 3: # C, H, W -> H, W, C
                    frame = frame.permute(1, 2, 0)
                if frame.dtype == torch.float32 or frame.dtype == torch.float16:
                     frame = (frame * 255).clamp(0, 255).to(torch.uint8)
                processed_frames.append(frame.numpy())
            video_frames = processed_frames

        # 3. Manually save with High Quality settings
        imageio.mimsave(
            str(output_path),
            video_frames,
            fps=24,
            plugin='ffmpeg',
            ffmpeg_params=[
                '-crf', '18',          
                '-preset', 'slow',    
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p' 
            ]
        )
        
        _ = time.time() - start

        if not output_path.exists():
            raise RuntimeError(f"Generation completed but output not found: {output_path}")
        if output_path.stat().st_size < 10_000:
            raise RuntimeError(f"Generated file too small ({output_path.stat().st_size} bytes): {output_path}")

        return str(output_path)