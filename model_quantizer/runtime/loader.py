"""Load local raw snapshots or quantized artifacts for benchmark evaluation."""

from __future__ import annotations





import time
import torch
from pathlib      import Path
from dataclasses  import dataclass
from typing       import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_quantizer.artifacts.loader import (
    QuantizedArtifactLoader,
    load_checkpoint_state_dict_with_timings,
    load_normalized_model_config,
    validate_state_dict_match,
)
from model_quantizer.configuration import ModelConfig, ProjectConfig
from model_quantizer.download.downloader import ModelDownloader
from model_quantizer.utils.device import resolve_compute_device, resolve_torch_dtype
from model_quantizer.utils.filesystem import sanitize_name


@dataclass(frozen=True)
class ModelLoadRequest:
    """Resolved inputs for loading one local model variant."""

    model_name:     str
    source:         str
    quantizer_name: Optional[str]
    device:         str


@dataclass(frozen=True)
class LoadedModelBundle:
    """All runtime objects needed for one benchmark evaluation pass."""

    model_name:      str
    source:          str
    source_path:     Path
    resolved_device: torch.device
    load_seconds:    float
    normalized_load_seconds: float
    load_breakdown:  Dict[str, Any]
    tokenizer:       Any
    model:           AutoModelForCausalLM


class LocalModelLoader:
    """Loads local raw snapshots or quantized artifacts for evaluation."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.downloader = ModelDownloader(
            config.paths.raw_models_dir,
            prefer_safetensors=config.runtime.prefer_safetensors,
        )

    def load(self, request: ModelLoadRequest) -> LoadedModelBundle:
        """Load one local model variant from disk."""

        try:
            model_config = self.config.models[request.model_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.config.models))
            raise ValueError(
                f"Unknown model '{request.model_name}'. Available models: {available}"
            ) from exc

        resolved_device = resolve_compute_device(request.device)
        load_start = time.perf_counter()

        if request.source == "raw":
            source_path, tokenizer, model, load_breakdown = self._load_raw_model(
                model_config,
                resolved_device,
            )
        elif request.source == "quantized":
            if not request.quantizer_name:
                raise ValueError("quantizer_name is required when loading a quantized artifact.")
            source_path, tokenizer, model, load_breakdown = self._load_quantized_model(
                request.model_name,
                request.quantizer_name,
                resolved_device,
            )
        else:
            raise ValueError(f"Unsupported model source '{request.source}'.")

        self._prepare_tokenizer(tokenizer)
        model.config.use_cache = False
        model.eval()
        total_load_seconds = time.perf_counter() - load_start
        load_breakdown["end_to_end_load_seconds"] = round(total_load_seconds, 6)
        load_breakdown["normalized_load_seconds"] = round(
            self._normalized_load_seconds(load_breakdown),
            6,
        )

        return LoadedModelBundle(
            model_name      = request.model_name,
            source          = request.source,
            source_path     = source_path,
            resolved_device = resolved_device,
            load_seconds    = total_load_seconds,
            normalized_load_seconds = float(load_breakdown["normalized_load_seconds"]),
            load_breakdown  = load_breakdown,
            tokenizer       = tokenizer,
            model           = model,
        )

    def _load_raw_model(
        self,
        model_config: ModelConfig,
        resolved_device: torch.device,
    ) -> tuple[Path, Any, AutoModelForCausalLM, Dict[str, Any]]:
        downloaded = self.downloader.require_local_snapshot(model_config)
        tokenizer_start = time.perf_counter()
        tokenizer  = AutoTokenizer.from_pretrained(
            downloaded.local_path,
            trust_remote_code = model_config.trust_remote_code,
            local_files_only  = True,
        )
        tokenizer_seconds = time.perf_counter() - tokenizer_start

        config_start = time.perf_counter()
        config = load_normalized_model_config(
            downloaded.local_path,
            trust_remote_code = model_config.trust_remote_code,
        )
        config_seconds = time.perf_counter() - config_start

        state_dict_result = load_checkpoint_state_dict_with_timings(downloaded.local_path)
        torch_dtype = self._resolve_model_dtype(model_config, resolved_device)
        model, build_breakdown = self._build_model_from_state_dict(
            state_dict=state_dict_result.state_dict,
            config=config,
            trust_remote_code=model_config.trust_remote_code,
            resolved_device=resolved_device,
            target_dtype=torch_dtype,
        )
        load_breakdown = {
            "load_strategy": "state_dict_reconstruction",
            "source_kind": "raw",
            "tokenizer_load_seconds": round(tokenizer_seconds, 6),
            "config_load_seconds": round(config_seconds, 6),
            "manifest_load_seconds": 0.0,
            "dequantize_seconds": 0.0,
            **state_dict_result.phase_seconds,
            **build_breakdown,
        }
        return downloaded.local_path, tokenizer, model, load_breakdown

    def _load_quantized_model(
        self,
        model_name: str,
        quantizer_name: str,
        resolved_device: torch.device,
    ) -> tuple[Path, Any, AutoModelForCausalLM, Dict[str, Any]]:
        artifact_dir = (
            self.config.paths.quantized_models_dir
            / sanitize_name(model_name)
            / sanitize_name(quantizer_name)
        )
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Quantized artifact directory not found: {artifact_dir}. "
                "Run the quantization pipeline first."
            )

        manifest_start = time.perf_counter()
        manifest = QuantizedArtifactLoader.load_manifest(artifact_dir)
        manifest_seconds = time.perf_counter() - manifest_start

        tokenizer_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            artifact_dir,
            trust_remote_code=bool(manifest["source_model"]["trust_remote_code"]),
            local_files_only=True,
        )
        tokenizer_seconds = time.perf_counter() - tokenizer_start

        config_start = time.perf_counter()
        config = load_normalized_model_config(
            artifact_dir,
            trust_remote_code=bool(manifest["source_model"]["trust_remote_code"]),
        )
        config_seconds = time.perf_counter() - config_start

        state_dict_result = QuantizedArtifactLoader.load_state_dict_with_timings(
            artifact_dir,
            manifest=manifest,
        )
        original_dtype = self._resolve_model_dtype(
            self.config.models[model_name],
            resolved_device,
        )
        model, build_breakdown = self._build_model_from_state_dict(
            state_dict=state_dict_result.state_dict,
            config=config,
            trust_remote_code=bool(manifest["source_model"]["trust_remote_code"]),
            resolved_device=resolved_device,
            target_dtype=original_dtype,
        )
        load_breakdown = {
            "load_strategy": "state_dict_reconstruction",
            "source_kind": "quantized",
            "tokenizer_load_seconds": round(tokenizer_seconds, 6),
            "config_load_seconds": round(config_seconds, 6),
            "manifest_load_seconds": round(manifest_seconds, 6),
            **state_dict_result.phase_seconds,
            **build_breakdown,
        }
        return artifact_dir, tokenizer, model, load_breakdown

    @staticmethod
    def _build_model_from_state_dict(
        *,
        state_dict: Dict[str, torch.Tensor],
        config,
        trust_remote_code: bool,
        resolved_device: torch.device,
        target_dtype: Optional[torch.dtype],
    ) -> tuple[AutoModelForCausalLM, Dict[str, float]]:
        """Construct a dense model from config and apply a provided state dict."""

        build_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if getattr(config, "model_type", None) == "phi3":
            build_kwargs["attn_implementation"] = "eager"

        model_init_start = time.perf_counter()
        model = AutoModelForCausalLM.from_config(
            config,
            **build_kwargs,
        )
        model_init_seconds = time.perf_counter() - model_init_start

        apply_start = time.perf_counter()
        incompatible = model.load_state_dict(state_dict, strict=False)
        apply_seconds = time.perf_counter() - apply_start
        validate_state_dict_match(model, config, incompatible)

        dtype_cast_seconds = 0.0
        if target_dtype is not None:
            dtype_start = time.perf_counter()
            model = model.to(dtype=target_dtype)
            dtype_cast_seconds = time.perf_counter() - dtype_start

        transfer_start = time.perf_counter()
        model = model.to(resolved_device)
        transfer_seconds = time.perf_counter() - transfer_start

        return model, {
            "model_init_seconds": round(model_init_seconds, 6),
            "state_dict_apply_seconds": round(apply_seconds, 6),
            "dtype_cast_seconds": round(dtype_cast_seconds, 6),
            "device_transfer_seconds": round(transfer_seconds, 6),
        }

    @staticmethod
    def _normalized_load_seconds(load_breakdown: Dict[str, Any]) -> float:
        """Load time for the shared reconstruction path, excluding tokenizer/config setup."""

        total = 0.0
        for key in (
            "manifest_load_seconds",
            "weight_read_seconds",
            "weight_decode_seconds",
            "model_init_seconds",
            "state_dict_apply_seconds",
            "dtype_cast_seconds",
            "device_transfer_seconds",
        ):
            value = load_breakdown.get(key)
            if value is None:
                continue
            total += float(value)
        return total

    @staticmethod
    def _prepare_tokenizer(tokenizer: Any) -> None:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    @staticmethod
    def _resolve_model_dtype(
        model_config: ModelConfig,
        resolved_device: torch.device,
    ) -> Optional[torch.dtype]:
        dtype = resolve_torch_dtype(model_config.torch_dtype)
        if resolved_device.type == "cpu" and dtype == torch.float16:
            return torch.float32
        return dtype


def build_prompt_text(
    tokenizer: Any,
    system_prompt: Optional[str],
    user_prompt: str,
    *,
    use_chat_template: bool = False,
) -> str:
    """Render a benchmark prompt for likelihood scoring."""

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt})
    return _render_prompt(
        tokenizer,
        messages,
        use_chat_template = use_chat_template,
    )


def _render_prompt(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    use_chat_template: bool,
) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if use_chat_template and chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    lines: List[str] = []
    role_names = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    for message in messages:
        role = role_names.get(message["role"], message["role"].capitalize())
        lines.append(f"{role}: {message['content'].strip()}")
    lines.append("Assistant:")
    return "\n\n".join(lines)
