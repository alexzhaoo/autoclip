import json
import inspect
import os
import random
import socket
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch


def _log_cuda_memory(tag: str) -> None:
    print(f"[CUDA] {tag}: logger called", flush=True)
    # Default ON (prints only a couple lines during init). Opt-out with WAN22_DEBUG_CUDA=0/false.
    if os.getenv("WAN22_DEBUG_CUDA", "1").strip().lower() in {"0", "false", "no", "n"}:
        print(f"[CUDA] {tag}: disabled via WAN22_DEBUG_CUDA", flush=True)
        return
    if not torch.cuda.is_available():
        print(f"[CUDA] {tag}: cuda not available", flush=True)
        return
    try:
        # Force lazy CUDA init so mem_get_info is meaningful.
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


def _probe_first_param_device(obj: object) -> None:
    """Best-effort probe to see where weights actually live (CPU vs CUDA)."""

    try:
        import torch.nn as nn
    except Exception:
        return

    seen: set[int] = set()

    def walk(o: object, depth: int) -> Optional["torch.nn.Module"]:
        if depth <= 0:
            return None
        oid = id(o)
        if oid in seen:
            return None
        seen.add(oid)

        if isinstance(o, nn.Module):
            return o

        d = getattr(o, "__dict__", None)
        if not isinstance(d, dict):
            return None

        for v in d.values():
            if isinstance(v, (str, int, float, bool, type(None))):
                continue
            m = walk(v, depth - 1)
            if m is not None:
                return m
        return None

    module = walk(obj, depth=6)
    if module is None:
        print("[CUDA] probe: could not find torch.nn.Module in pipeline", flush=True)
        return

    try:
        p = next(module.parameters())
        print(f"[CUDA] probe: first parameter device={p.device}, dtype={p.dtype}", flush=True)
    except StopIteration:
        print("[CUDA] probe: module has no parameters", flush=True)
    except Exception as e:
        print(f"[CUDA] probe: failed ({type(e).__name__}: {e})", flush=True)


def _pipe_create_generator(pipe: object, **kwargs) -> None:
    """Call LightX2V create_generator with best-effort device placement."""

    create_fn = getattr(pipe, "create_generator")
    if not torch.cuda.is_available():
        create_fn(**kwargs)
        return

    # LightX2V's API has changed across versions. In some builds, signature
    # introspection fails (or create_generator is wrapped), so we can't reliably
    # detect supported kwargs. To keep the runner from initializing on CPU,
    # we optimistically pass CUDA placement hints and fall back if rejected.
    candidate_kwargs = dict(kwargs)
    candidate_kwargs.setdefault("device", "cuda")
    candidate_kwargs.setdefault("device_id", 0)
    candidate_kwargs.setdefault("local_rank", 0)

    try:
        create_fn(**candidate_kwargs)
        return
    except TypeError:
        # Retry after removing any unsupported placement kwargs.
        # This keeps compatibility with older LightX2V versions.
        stripped = dict(kwargs)
        create_fn(**stripped)
        return


def _brute_force_move_lightx2v_to_cuda(pipe: object) -> None:
    """Best-effort move of LightX2V internals to CUDA.

    This is intentionally defensive: LightX2V object graphs vary by version.
    """

    if not torch.cuda.is_available():
        print("[WAN22] BRUTE FORCE: CUDA not available", flush=True)
        return

    print("[WAN22] BRUTE FORCE: attempting to move pipeline internals to CUDA...", flush=True)

    # 1) Try the wrapper itself.
    to_fn = getattr(pipe, "to", None)
    if callable(to_fn):
        try:
            to_fn("cuda")
            print("[WAN22] BRUTE FORCE: pipe.to('cuda') succeeded", flush=True)
        except Exception as e:
            print(f"[WAN22] BRUTE FORCE: pipe.to('cuda') failed: {e}", flush=True)

    # 2) Try common internal model attribute names.
    possible_model_attrs = ["model", "transformer", "dit", "unet", "denoiser", "runner", "generator"]
    moved_any = False
    for attr in possible_model_attrs:
        try:
            if not hasattr(pipe, attr):
                continue
            sub = getattr(pipe, attr)
            sub_to = getattr(sub, "to", None)
            if callable(sub_to):
                print(f"[WAN22] BRUTE FORCE: Found '{attr}', moving to CUDA...", flush=True)
                sub_to("cuda")
                moved_any = True
        except Exception as e:
            print(f"[WAN22] BRUTE FORCE: moving '{attr}' failed: {e}", flush=True)

    if not moved_any:
        print("[WAN22] BRUTE FORCE: WARNING: did not find a known internal model attr to move", flush=True)

    # 3) Verification probe.
    _probe_first_param_device(pipe)


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


def _maybe_patch_lightx2v_config_json(
    config_json: Union[str, Path],
    distill_config: "Wan22DistillConfig",
) -> Optional[str]:
    """Ensure LightX2V distill config has keys required by current runner.

    Some LightX2V releases assume certain keys exist in the config JSON. In
    single-process usage, users often point WAN22_LIGHTX2V_CONFIG_JSON at a
    generic config which may omit distill-specific fields.

    Returns a path to a patched temp JSON (string) or None if patching failed.
    """

    try:
        path = Path(config_json)
        raw = path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
        if not isinstance(cfg, dict):
            return None

        # Required by Wan22StepDistillScheduler in LightX2V.
        cfg.setdefault("denoising_step_list", list(distill_config.denoising_step_list))

        # These are commonly consumed; safe defaults keep behavior consistent.
        cfg.setdefault("boundary_step_index", distill_config.boundary_step_index)
        cfg.setdefault("infer_steps", distill_config.num_inference_steps)
        cfg.setdefault("guidance_scale", distill_config.guidance_scale)
        cfg.setdefault("sample_shift", distill_config.sample_shift)
        cfg.setdefault("height", distill_config.height)
        cfg.setdefault("width", distill_config.width)
        cfg.setdefault("num_frames", distill_config.num_frames)

        fd, temp_path = tempfile.mkstemp(prefix="wan22_lightx2v_cfg_", suffix=".json")
        os.close(fd)
        Path(temp_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return temp_path
    except Exception:
        return None


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
        offload_model: bool = False,  # Default: keep on GPU
        config: Wan22DistillConfig = Wan22DistillConfig(),
        attn_mode: str = "flash_attn2",
    ):
        print("[WAN22] Entering Wan22LightX2VGenerator.__init__", flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Wan2.2 generation")

        if config.num_inference_steps != 4:
            raise ValueError("Wan2.2 4-step distilled LoRAs require num_inference_steps=4")

        os.environ.setdefault("DTYPE", "BF16")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Explicit override: on high-VRAM GPUs we want everything resident on GPU.
        # If set, this forces offload_model=False even if a caller passes True.
        if os.getenv("WAN22_FORCE_GPU", "").strip().lower() in {"1", "true", "yes", "y"}:
            offload_model = False

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

        # Aggressive opt-in: attempt to place weights on GPU as early as possible.
        # Note: depending on LightX2V version, weights may still be loaded during
        # create_generator(); in that case, the most important part is passing
        # device hints via _pipe_create_generator().
        # NEW (Always runs if offload is disabled):
        if not offload_model:
             print("[WAN22] Running Brute Force Move...", flush=True) # Optional debug print
             _brute_force_move_lightx2v_to_cuda(self.pipe)
        _log_cuda_memory("after LightX2VPipeline()")

        if offload_model:
            self.pipe.enable_offload(
                cpu_offload=True,
                offload_granularity="model",
                text_encoder_offload=False,
                image_encoder_offload=False,
                vae_offload=False,
            )

        # NOTE: LightX2V appears to lazily materialize/load the actual model
        # (torch.nn.Module + weights) during create_generator() in some versions.
        # To ensure LoRA patching happens on GPU, we create the generator first,
        # then move the now-materialized model to CUDA, then apply LoRAs.
        high_path = str(self.high_noise_lora_path)
        low_path = str(self.low_noise_lora_path)

        attn_mode = self._maybe_downgrade_attn_mode(attn_mode)

        # Prefer LightX2V's own reference config when available.
        # Grainy/under-denoised outputs are often caused by mismatched distill settings.
        effective_config_json = self._resolve_lightx2v_config_json(config)

        # LightX2V currently calls torch.distributed.get_rank() during generator creation.
        _ensure_torch_distributed_initialized()

        if effective_config_json is not None:
            patched = _maybe_patch_lightx2v_config_json(effective_config_json, config)
            if patched is not None:
                print(f"[INFO] LightX2V generator config_json: {effective_config_json} (patched: {patched})")
                _pipe_create_generator(self.pipe, config_json=str(patched))
            else:
                print(
                    f"[WARN] LightX2V generator config_json could not be patched ({effective_config_json}); "
                    "falling back to inline distill params"
                )
                _pipe_create_generator(
                    self.pipe,
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
        else:
            print("[INFO] LightX2V generator config_json: (none) using inline distill params")
            _pipe_create_generator(
                self.pipe,
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

        # Now that create_generator() has (likely) materialized the underlying model,
        # force GPU residency before applying LoRAs.
        brute_force = os.getenv("WAN22_BRUTE_FORCE_CUDA", "").strip().lower() in {"1", "true", "yes", "y"}
        force_gpu = os.getenv("WAN22_FORCE_GPU", "").strip().lower() in {"1", "true", "yes", "y"}
        if not offload_model and (force_gpu or brute_force):
            _brute_force_move_lightx2v_to_cuda(self.pipe)

        # Apply Dual-LoRA after the base weights exist and (optionally) are on CUDA.
        # LightX2V's LoRA API and expected LoRA names have changed across versions.
        # For Wan2.2 distill runners, LightX2V requires explicit LoRA names.
        self.pipe.enable_lora(
            [
                {"name": "high_noise_model", "path": high_path, "strength": 1.0},
                {"name": "low_noise_model", "path": low_path, "strength": 1.0},
            ]
        )

        _log_cuda_memory("after create_generator()")
        _probe_first_param_device(self.pipe)

        print("[WAN22] Finished Wan22LightX2VGenerator.__init__", flush=True)

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
        try:
            self.pipe.generate(
                seed=seed,
                prompt=enhanced_prompt,
                negative_prompt="",
                save_result_path=str(output_path),
            )
        except Exception as e:
            import traceback
            from importlib import metadata

            lightx2v_version = None
            try:
                lightx2v_version = metadata.version("lightx2v")
            except Exception:
                lightx2v_version = None

            # The most common cause of "'NoneType' object is not callable" here is a
            # LightX2V internal callback/hook (often LoRA switching in Wan distill runners)
            # not being initialized due to a version/API mismatch.
            diag = {
                "lightx2v_version": lightx2v_version,
                "pipe_type": type(self.pipe).__name__,
                "has_create_generator": callable(getattr(self.pipe, "create_generator", None)),
                "has_enable_lora": callable(getattr(self.pipe, "enable_lora", None)),
                "has_generate": callable(getattr(self.pipe, "generate", None)),
                "models_dir": str(self.models_dir),
                "base_model_path": str(self.base_model_path),
                "high_noise_lora_path": str(self.high_noise_lora_path),
                "low_noise_lora_path": str(self.low_noise_lora_path),
            }

            tb = traceback.format_exc()
            raise RuntimeError(
                "LightX2V generation failed inside LightX2VPipeline.generate(). "
                "If the error is 'NoneType object is not callable', it is usually caused by "
                "a LightX2V version mismatch where the Wan distill runner expects a hook/callback "
                "(often LoRA switching HIGH/LOW noise models) that was not initialized. "
                f"Original error: {type(e).__name__}: {e}\n"
                f"Diagnostics: {diag}\n"
                f"Traceback:\n{tb}"
            ) from e
        _ = time.time() - start

        if not output_path.exists():
            raise RuntimeError(f"Generation completed but output not found: {output_path}")
        if output_path.stat().st_size < 10_000:
            raise RuntimeError(f"Generated file too small ({output_path.stat().st_size} bytes): {output_path}")

        return str(output_path)
