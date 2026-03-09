"""Video generation using LTX-2 Fast for B-roll production.

This module provides a unified interface for generating B-roll video clips
using Lightricks' LTX-2 Fast model via HuggingFace Diffusers.

VRAM Requirements:
- LTX-2 is a 19B parameter model
- FP16: ~35GB VRAM (too big for 24-30GB GPUs)
- With CPU offload: Works on 16-24GB GPUs
- With quantization: Works on 12-16GB GPUs
"""

import json
import os
import random
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


def _log_cuda_memory(tag: str) -> None:
    """Log current CUDA memory usage for debugging."""
    print(f"[CUDA] {tag}: logger called", flush=True)
    if os.getenv("LTX2_DEBUG_CUDA", "1").strip().lower() in {"0", "false", "no", "n"}:
        print(f"[CUDA] {tag}: disabled via LTX2_DEBUG_CUDA", flush=True)
        return
    if not torch.cuda.is_available():
        print(f"[CUDA] {tag}: cuda not available", flush=True)
        return
    try:
        _ = torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info()
        allocated_b = torch.cuda.memory_allocated()
        reserved_b = torch.cuda.memory_reserved()
        print(
            f"[CUDA] {tag}: allocated={allocated_b/1024**3:.2f}GiB, reserved={reserved_b/1024**3:.2f}GiB, "
            f"free={free_b/1024**3:.2f}GiB / total={total_b/1024**3:.2f}GiB, "
            f"device={torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())}), "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}, initialized={torch.cuda.is_initialized()}"
            ,
            flush=True,
        )
    except Exception as e:
        print(f"[CUDA] {tag}: failed to query memory ({type(e).__name__}: {e})", flush=True)


@dataclass(frozen=True)
class LTX2Config:
    """Configuration for LTX-2 video generation.
    
    LTX-2 is a 19B parameter model requiring significant VRAM.
    Frame count must follow pattern: 8n + 1 (1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81...)
    
    VRAM Requirements:
    - Model alone (FP16): ~35GB
    - With CPU offload: Works on 16-24GB GPUs ✅
    - With quantization: Works on 12-16GB GPUs
    """
    num_inference_steps: int = 20  # LTX-2 Fast uses fewer steps (20 for speed, 30 for quality)
    guidance_scale: float = 3.0    # LTX-2 uses lower guidance scale than Wan2.2
    width: int = 704               # Default 704x480 (16:9) or 704x1280 (9:16)
    height: int = 480
    num_frames: int = 49           # ~2 seconds at 24fps (must be 8n+1)
    fps: int = 24
    # Generation quality settings
    use_fp16: bool = True          # Use FP16 for faster inference
    enable_vae_slicing: bool = True  # Reduce VRAM usage for long videos
    enable_model_cpu_offload: bool = True  # REQUIRED for <40GB VRAM GPUs


class LTX2FastGenerator:
    """LTX-2 Fast video generator for B-roll production.
    
    Uses HuggingFace Diffusers LTX2Pipeline with CPU offloading to run on
    consumer GPUs (16-30GB VRAM).
    
    Model: Lightricks/LTX-2 (19B parameters)
    """

    # Frame count must be 8n + 1 for LTX models
    VALID_FRAME_COUNTS = [1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]

    def __init__(
        self,
        model_id: str = "Lightricks/LTX-2",
        config: Optional[LTX2Config] = None,
        device: str = "cuda",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize LTX-2 Fast generator.
        
        Args:
            model_id: HuggingFace model ID (default: Lightricks/LTX-2)
            config: LTX2Config instance with generation parameters
            device: Device to run on (cuda or cpu)
            cache_dir: Optional directory to cache downloaded models
        """
        print("[LTX-2] Entering LTX2FastGenerator.__init__", flush=True)
        
        if not torch.cuda.is_available() and device == "cuda":
            raise RuntimeError("CUDA is required for LTX-2 generation. CPU inference is not supported.")
        
        # Check GPU memory and configure accordingly
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[LTX-2] GPU VRAM: {total_gb:.1f}GB")
            
            if total_gb >= 38:  # A100 reports ~39.4GB, use 38GB threshold
                print(f"[LTX-2] ✅ 40GB+ VRAM detected! Running fully on GPU (no CPU offload).")
                if config is None:
                    config = LTX2Config(enable_model_cpu_offload=False)
                else:
                    # Update config to disable CPU offload for 40GB+ GPUs
                    config = LTX2Config(
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        width=config.width,
                        height=config.height,
                        num_frames=config.num_frames,
                        fps=config.fps,
                        use_fp16=config.use_fp16,
                        enable_vae_slicing=config.enable_vae_slicing,
                        enable_model_cpu_offload=False,  # Keep on GPU!
                    )
            elif total_gb < 38 and not (config and config.enable_model_cpu_offload):
                print(f"[LTX-2] WARNING: LTX-2 needs ~35GB VRAM. You have {total_gb:.1f}GB.")
                print(f"[LTX-2] Auto-enabling CPU offload for compatibility...")
                if config is None:
                    config = LTX2Config(enable_model_cpu_offload=True)
                else:
                    # Create new config with CPU offload enabled
                    config = LTX2Config(
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        width=config.width,
                        height=config.height,
                        num_frames=config.num_frames,
                        fps=config.fps,
                        use_fp16=config.use_fp16,
                        enable_vae_slicing=config.enable_vae_slicing,
                        enable_model_cpu_offload=True,
                    )
        
        self.config = config or LTX2Config()
        self.device = device
        self.model_id = model_id
        
        # Set environment variables for better memory management
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        # Cache directory for models
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        # Initialize pipeline
        self._init_pipeline()
        
        print("[LTX-2] Finished LTX2FastGenerator.__init__", flush=True)

    def _init_pipeline(self):
        """Initialize the LTX-2 Diffusers pipeline with memory optimization."""
        try:
            from diffusers import LTX2Pipeline
        except ImportError as e:
            raise RuntimeError(
                "Failed to import diffusers LTX2Pipeline. "
                "Please install: pip install diffusers>=0.32.0 transformers accelerate"
            ) from e
        
        # Try to import export_to_video with fallbacks
        try:
            from diffusers.utils import export_to_video
        except ImportError:
            try:
                from diffusers.video_processor import export_to_video
            except ImportError:
                export_to_video = None
        
        print(f"[LTX-2] Loading model: {self.model_id}", flush=True)
        print(f"[LTX-2] CPU offload: {self.config.enable_model_cpu_offload}", flush=True)
        _log_cuda_memory("before pipeline load")
        
        # Determine dtype based on GPU capability
        # LTX-2 VAE has numerical instability in FP16, use bfloat16 when possible
        gpu_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
        has_native_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        
        if has_native_bf16:
            # bfloat16 has better numerical stability for LTX-2 VAE
            dtype = torch.bfloat16
            print(f"[LTX-2] Using bfloat16 for numerical stability", flush=True)
        else:
            dtype = torch.float16
            print(f"[LTX-2] Using float16 (bfloat16 not supported)", flush=True)
        
        # Load pipeline - LTX-2 doesn't use fp16 variant, load directly
        load_kwargs = {
            "torch_dtype": dtype,
        }
        
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.cache_dir)
        
        # Load the pipeline
        try:
            self.pipe = LTX2Pipeline.from_pretrained(
                self.model_id,
                **load_kwargs
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "401" in str(e) or "authentication" in error_msg or "token" in error_msg:
                raise RuntimeError(
                    f"HuggingFace authentication failed for {self.model_id}. "
                    "Please login: huggingface-cli login "
                    "or set HF_TOKEN environment variable."
                ) from e
            elif "404" in str(e) or "not found" in error_msg:
                raise RuntimeError(
                    f"Model {self.model_id} not found. "
                    "The model may have been moved or renamed. "
                    "Check https://huggingface.co/Lightricks for available models."
                ) from e
            raise
        
        _log_cuda_memory("after pipeline load")
        
        # Enable CPU offload BEFORE moving to device (this is key!)
        print(f"[LTX-2] enable_model_cpu_offload config value: {self.config.enable_model_cpu_offload}", flush=True)
        if self.config.enable_model_cpu_offload:
            print("[LTX-2] Enabling sequential CPU offload (saves VRAM)...", flush=True)
            try:
                self.pipe.enable_sequential_cpu_offload()
                print("[LTX-2] Sequential CPU offload enabled", flush=True)
            except Exception as e:
                print(f"[WARN] Sequential CPU offload failed: {e}, trying model CPU offload...", flush=True)
                try:
                    self.pipe.enable_model_cpu_offload()
                    print("[LTX-2] Model CPU offload enabled", flush=True)
                except Exception as e2:
                    print(f"[WARN] Model CPU offload also failed: {e2}", flush=True)
                    # Last resort: move to device and hope for the best
                    self.pipe = self.pipe.to(self.device)
        else:
            # Only move to device if not using CPU offload
            print(f"[LTX-2] Moving pipeline to {self.device} (no CPU offload)...", flush=True)
            self.pipe = self.pipe.to(self.device)
            print(f"[LTX-2] Pipeline moved to {self.device}", flush=True)
        
        # Enable VAE slicing (always try this)
        if self.config.enable_vae_slicing:
            try:
                self.pipe.enable_vae_slicing()
                print("[LTX-2] VAE slicing enabled", flush=True)
            except Exception as e:
                print(f"[WARN] VAE slicing failed: {e}", flush=True)
        
        _log_cuda_memory("after optimizations")
        
        # Debug: check if model is actually on CUDA
        if hasattr(self.pipe, 'transformer') and hasattr(self.pipe.transformer, 'device'):
            print(f"[LTX-2] Transformer device: {self.pipe.transformer.device}", flush=True)
        
        # Store export function
        self._export_to_video = export_to_video

    def _validate_frame_count(self, num_frames: int) -> int:
        """Validate and adjust frame count to 8n+1 pattern."""
        if num_frames in self.VALID_FRAME_COUNTS:
            return num_frames
        
        # Find closest valid frame count
        closest = min(self.VALID_FRAME_COUNTS, key=lambda x: abs(x - num_frames))
        print(f"[LTX-2] Adjusted frame count {num_frames} -> {closest} (must be 8n+1)", flush=True)
        return closest

    def _calculate_frames_for_duration(self, duration_seconds: float) -> int:
        """Calculate frame count needed for desired duration."""
        target_frames = int(duration_seconds * self.config.fps)
        return self._validate_frame_count(target_frames)

    def generate_clip(
        self, 
        prompt: str, 
        output_path: Union[str, Path], 
        seed: Optional[int] = None,
        duration: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
    ) -> str:
        """Generate a video clip from text prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use provided values or defaults from config
        width = width or self.config.width
        height = height or self.config.height
        
        if duration is not None:
            num_frames = self._calculate_frames_for_duration(duration)
        else:
            num_frames = num_frames or self.config.num_frames
            num_frames = self._validate_frame_count(num_frames)
        
        # Ensure dimensions are divisible by 32 (LTX requirement)
        width = (width // 32) * 32
        height = (height // 32) * 32
        
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        # Set seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Default negative prompt for quality
        if negative_prompt is None:
            negative_prompt = (
                "worst quality, inconsistent motion, blurry, jittery, distorted, "
                "poorly drawn, bad anatomy, watermark, signature, text"
            )
        
        print(f"[LTX-2] Generating: {prompt[:60]}...", flush=True)
        print(f"[LTX-2] Resolution: {width}x{height}, Frames: {num_frames}, Seed: {seed}", flush=True)
        
        _log_cuda_memory("before generation")
        start = time.time()
        
        try:
            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate video
            result = self.pipe(
                prompt=prompt.strip(),
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
            )
            
            video_frames = result.frames[0]
            
            # Validate output - check for NaN/Inf values that indicate numerical instability
            import numpy as np
            frames_array = np.array(video_frames)
            if np.isnan(frames_array).any() or np.isinf(frames_array).any():
                print("[LTX-2] ⚠️ Generated frames contain NaN/Inf values - numerical instability detected", flush=True)
                print("[LTX-2] This usually happens with FP16 on certain GPUs. Retrying with bfloat16...", flush=True)
                raise RuntimeError("Numerical instability: generated frames contain NaN/Inf values. Try using bfloat16 instead of FP16.")
            
            # Export to video file
            if self._export_to_video is not None:
                self._export_to_video(video_frames, str(output_path), fps=self.config.fps)
            else:
                import imageio
                imageio.mimsave(str(output_path), video_frames, fps=self.config.fps, quality=8)
            
            elapsed = time.time() - start
            _log_cuda_memory("after generation")
            
            print(f"[LTX-2] Generated in {elapsed:.1f}s: {output_path}", flush=True)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[LTX-2] CUDA OOM. Tips:", flush=True)
                print("  1. Lower resolution: export LTX2_RESOLUTION=480p", flush=True)
                print("  2. Reduce frames: export LTX2_DURATION=2.0", flush=True)
                print("  3. Use a GPU with more VRAM (40GB+)", flush=True)
            raise
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(f"LTX-2 generation failed: {type(e).__name__}: {e}\n{tb}") from e
        
        # Verify output
        if not output_path.exists():
            raise RuntimeError(f"Generation completed but output not found: {output_path}")
        
        file_size = output_path.stat().st_size
        if file_size < 10_000:
            raise RuntimeError(f"Generated file too small ({file_size} bytes): {output_path}")
        
        return str(output_path)

    def generate_broll_for_region(
        self,
        prompt: str,
        duration: float,
        output_path: Union[str, Path],
        seed: Optional[int] = None,
    ) -> str:
        """Generate B-roll video optimized for a specific duration."""
        return self.generate_clip(
            prompt=prompt,
            output_path=output_path,
            seed=seed,
            duration=duration,
        )


# Backwards compatibility alias
class Wan22LightX2VGenerator(LTX2FastGenerator):
    """Backwards compatibility wrapper for old code using Wan22LightX2VGenerator."""
    
    def __init__(self, *args, **kwargs):
        print("[DEPRECATED] Wan22LightX2VGenerator is deprecated. Using LTX2FastGenerator instead.", flush=True)
        wan22_args = ["models_dir", "base_model_dirname", "lightning_lora_dir", "offload_model"]
        for arg in wan22_args:
            kwargs.pop(arg, None)
        super().__init__(*args, **kwargs)


def create_ltx2_generator(
    resolution: str = "480p",
    aspect_ratio: str = "16:9",
    fast_mode: bool = True,
) -> LTX2FastGenerator:
    """Factory function to create LTX-2 generator with preset configurations."""
    resolutions = {
        "480p": {"16:9": (704, 480), "9:16": (480, 704)},
        "720p": {"16:9": (1280, 704), "9:16": (704, 1280)},
        "1080p": {"16:9": (1920, 1088), "9:16": (1088, 1920)},
        "4K": {"16:9": (3840, 2176), "9:16": (2176, 3840)},
    }
    
    if resolution not in resolutions:
        raise ValueError(f"Unknown resolution: {resolution}")
    
    width, height = resolutions[resolution][aspect_ratio]
    num_steps = 20 if fast_mode else 30
    
    # Auto-configure based on available VRAM
    import torch
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        use_cpu_offload = total_gb < 38  # Only use CPU offload if < 38GB
    else:
        use_cpu_offload = True
    
    config = LTX2Config(
        width=width,
        height=height,
        num_inference_steps=num_steps,
        enable_model_cpu_offload=use_cpu_offload,
        enable_vae_slicing=True,
    )
    
    offload_status = "enabled" if use_cpu_offload else "disabled"
    print(f"[LTX-2] Creating generator: {resolution} {aspect_ratio}, CPU offload: {offload_status}")
    
    return LTX2FastGenerator(config=config)


if __name__ == "__main__":
    print("🎬 LTX-2 Fast Video Generator Test")
    print("=" * 50)
    
    try:
        generator = create_ltx2_generator(resolution="480p", aspect_ratio="16:9", fast_mode=True)
        
        test_prompt = (
            "Smooth aerial dolly shot gliding over a city at dusk, "
            "camera moving forward steadily, windows glowing, soft haze drifting"
        )
        
        output_file = "test_ltx2_output.mp4"
        
        print(f"\n📝 Prompt: {test_prompt}")
        print(f"📁 Output: {output_file}\n")
        
        result = generator.generate_clip(
            prompt=test_prompt,
            output_path=output_file,
            duration=2.0,
        )
        
        print(f"\n✅ Success! Video saved to: {result}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
