import os
import random
import socket
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


def _ensure_torch_distributed_initialized() -> None:
    """LightX2V calls torch.distributed.get_rank() during init.

    In non-distributed, single-process runs (common for this repo), torch.distributed
    is available but not initialized, and get_rank() raises.
    """

    try:
        import torch.distributed as dist  # type: ignore
    except Exception:
        return

    if not dist.is_available() or dist.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Prefer env:// if the user is explicitly running distributed.
    if os.getenv("RANK") is not None or os.getenv("WORLD_SIZE") is not None:
        dist.init_process_group(backend=backend, init_method="env://")
        return

    # Otherwise initialize a local, single-process group.
    # Use a random free port to avoid collisions.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _addr, port = sock.getsockname()
    sock.close()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    # Some backends use the temp dir for rendezvous artifacts.
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(tempfile.gettempdir(), "torch_extensions"))

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )


@dataclass(frozen=True)
class Wan22DistillConfig:
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    width: int = 832
    height: int = 480
    num_frames: int = 81
    sample_shift: float = 5.0
    boundary_step_index: int = 2
    denoising_step_list: tuple[int, int, int, int] = (1000, 750, 500, 250)
    config_json: Optional[Union[str, Path]] = None
    rope_type: str = "torch"


class Wan22LightX2VGenerator:
    """LightX2V wrapper for Wan2.2-Lightning (4-step) Dual-LoRA switching.

    Wan2.2-Lightning LoRA releases ship as a directory containing:
    - high_noise_model.safetensors
    - low_noise_model.safetensors
    """

    @staticmethod
    def _resolve_lightning_lora_dir(
        models_dir: Path,
        lightning_lora_dir: Optional[Union[str, Path]],
        lightning_folder: str,
    ) -> Path:
        if lightning_lora_dir is not None:
            candidate = Path(lightning_lora_dir)
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"lightning_lora_dir does not exist: {candidate}")

        env_dir = os.getenv("WAN22_LIGHTNING_LORA_DIR")
        if env_dir:
            candidate = Path(env_dir)
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"WAN22_LIGHTNING_LORA_DIR does not exist: {candidate}")

        repo_root = Path(__file__).resolve().parent
        candidates = [
            models_dir / "loras" / lightning_folder,
            repo_root / "Wan2.2-Lightning" / lightning_folder,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Wan2.2-Lightning LoRA directory not found. "
            "Set WAN22_LIGHTNING_LORA_DIR or pass lightning_lora_dir, e.g. "
            "./Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0"
        )

    @staticmethod
    def _maybe_downgrade_attn_mode(attn_mode: str) -> str:
        if not attn_mode.startswith("flash_attn"):
            return attn_mode

        try:
            import flash_attn  # type: ignore[import-not-found]  # noqa: F401
        except Exception:
            return "sdpa"

        return attn_mode

    @staticmethod
    def _resolve_lightx2v_config_json(config: Wan22DistillConfig) -> Optional[Union[str, Path]]:
        if config.config_json is not None:
            return config.config_json

        env_cfg = os.getenv("WAN22_LIGHTX2V_CONFIG_JSON")
        if env_cfg:
            return env_cfg

        repo_cfg = (
            Path(__file__).resolve().parent
            / "LightX2V"
            / "configs"
            / "dist_infer"
            / "wan22_moe_t2v_cfg.json"
        )
        if repo_cfg.exists():
            return repo_cfg

        return None

    

    def __init__(
        self,
        models_dir: Union[str, Path] = "./models",
        base_model_dirname: str = "Wan2.2-T2V-A14B",
        lightning_lora_dir: Optional[Union[str, Path]] = None,
        offload_model: bool = False,  # Changed default to False to keep on GPU
        config: Wan22DistillConfig = Wan22DistillConfig(),
        attn_mode: str = "flash_attn2",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Wan2.2 generation")

        if config.num_inference_steps != 4:
            raise ValueError("Wan2.2 4-step distilled LoRAs require num_inference_steps=4")

        os.environ.setdefault("DTYPE", "BF16")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        self.models_dir = Path(models_dir)
        # Support both layouts:
        # - this repo's: ./models/Wan2.2-T2V-A14B
        # - README's:    ./Wan2.2-T2V-A14B
        base_model_candidate = Path(base_model_dirname)
        if base_model_candidate.exists():
            self.base_model_path = base_model_candidate
        else:
            repo_root = Path(__file__).resolve().parent
            preferred = self.models_dir / base_model_dirname
            fallback = repo_root / base_model_dirname
            self.base_model_path = preferred if preferred.exists() else fallback

        # Wan2.2-Lightning layout: a directory with high_noise_model/low_noise_model.
        # Default search order:
        # 1) explicit argument
        # 2) env var WAN22_LIGHTNING_LORA_DIR
        # 3) ./models/loras/<lightning folder>
        # 4) ./Wan2.2-Lightning/<lightning folder>
        lightning_folder = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0"
        self.lightning_lora_dir = self._resolve_lightning_lora_dir(
            models_dir=self.models_dir,
            lightning_lora_dir=lightning_lora_dir,
            lightning_folder=lightning_folder,
        )
        self.high_noise_lora_path = self.lightning_lora_dir / "high_noise_model.safetensors"
        self.low_noise_lora_path = self.lightning_lora_dir / "low_noise_model.safetensors"

        if not self.base_model_path.exists():
            raise FileNotFoundError(f"Base model not found at {self.base_model_path}")
        if not self.high_noise_lora_path.exists():
            raise FileNotFoundError(
                f"High-noise Lightning LoRA not found at {self.high_noise_lora_path}. "
                "Expected file name: high_noise_model.safetensors"
            )
        if not self.low_noise_lora_path.exists():
            raise FileNotFoundError(
                f"Low-noise Lightning LoRA not found at {self.low_noise_lora_path}. "
                "Expected file name: low_noise_model.safetensors"
            )

        try:
            from lightx2v import LightX2VPipeline  # type: ignore[import-not-found]
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

        attn_mode = self._maybe_downgrade_attn_mode(attn_mode)

        # Prefer LightX2V's own reference config when available.
        # Grainy/under-denoised outputs are often caused by mismatched distill settings.
        effective_config_json = self._resolve_lightx2v_config_json(config)

        # LightX2V currently calls torch.distributed.get_rank() during generator creation.
        _ensure_torch_distributed_initialized()

        if effective_config_json is not None:
            print(f"[INFO] LightX2V generator config_json: {effective_config_json}")
            self.pipe.create_generator(config_json=str(effective_config_json))
        else:
            print("[INFO] LightX2V generator config_json: (none) using inline distill params")
            self.pipe.create_generator(
                attn_mode=attn_mode,
                rope_type=config.rope_type,
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

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        enhanced_prompt = f"{prompt.strip()}"
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        start = time.time()
        self.pipe.generate(
            seed=seed,
            prompt=enhanced_prompt,
            negative_prompt="",
            save_result_path=str(output_path),
        )
        _ = time.time() - start

        if not output_path.exists():
            raise RuntimeError(f"Generation completed but output not found: {output_path}")
        if output_path.stat().st_size < 10_000:
            raise RuntimeError(f"Generated file too small ({output_path.stat().st_size} bytes): {output_path}")

        return str(output_path)
