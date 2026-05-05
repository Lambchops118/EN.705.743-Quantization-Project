"""Publish local run outputs to optional remote storage."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
from model_quantizer.configuration import ProjectConfig, S3OutputConfig


@dataclass(frozen=True)
class OutputPublisher:
    """Maps local output files to an S3 bucket and uploads them on demand."""

    s3_config: S3OutputConfig
    path_roots: Dict[str, Path]

    def __post_init__(self) -> None:
        normalized = {name: root.resolve() for name, root in self.path_roots.items()}
        object.__setattr__(self, "path_roots", normalized)
        object.__setattr__(self, "_s3_client", None)

    @property
    def enabled(self) -> bool:
        return self.s3_config.enabled

    def reference_for(self, path: Path) -> str:
        """Return the remote URI when S3 is enabled, else the local path."""

        if not self.enabled:
            return str(path)

        key = self._key_for(path)
        if key is None:
            return str(path)
        return f"s3://{self.s3_config.bucket}/{key}"

    def upload_file(self, path: Path) -> Optional[str]:
        """Upload one existing file if S3 output is enabled."""

        if not self.enabled:
            return None

        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Cannot upload missing file: {resolved}")

        key = self._key_for(resolved)
        if key is None:
            raise ValueError(f"No S3 output mapping configured for path: {resolved}")

        extra_args = self._upload_extra_args(resolved)
        if extra_args:
            self._client().upload_file(
                str(resolved),
                self.s3_config.bucket,
                key,
                ExtraArgs=extra_args,
            )
        else:
            self._client().upload_file(
                str(resolved),
                self.s3_config.bucket,
                key,
            )
        return f"s3://{self.s3_config.bucket}/{key}"

    def upload_files(self, paths: Iterable[Path]) -> Dict[str, str]:
        """Upload multiple files and return a local-path to URI mapping."""

        uploaded: Dict[str, str] = {}
        for path in paths:
            uri = self.upload_file(path)
            if uri:
                uploaded[str(path)] = uri
        return uploaded

    def _key_for(self, path: Path) -> Optional[str]:
        resolved = path.resolve()
        for root_name, root_path in self.path_roots.items():
            try:
                relative = resolved.relative_to(root_path)
            except ValueError:
                continue
            parts = [self.s3_config.prefix, root_name, relative.as_posix()]
            return "/".join(part for part in parts if part)
        return None

    @staticmethod
    def _upload_extra_args(path: Path) -> Dict[str, str]:
        content_type, _ = mimetypes.guess_type(path.name)
        if not content_type:
            return {}
        return {"ContentType": content_type}

    def _client(self):
        client = getattr(self, "_s3_client", None)
        if client is not None:
            return client

        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on runtime install
            raise RuntimeError(
                "S3 output requires boto3. Install project dependencies again."
            ) from exc

        session = boto3.session.Session(region_name=self.s3_config.region)
        client  = session.client("s3", endpoint_url=self.s3_config.endpoint_url)
        object.__setattr__(self, "_s3_client", client)
        return client


def build_output_publisher(config: ProjectConfig) -> OutputPublisher:
    """Build the project-wide output publisher."""

    return OutputPublisher(
        s3_config  = config.output_s3,
        path_roots = {
            "logs":                 config.paths.logs_dir,
            "artifacts/metadata":   config.paths.metadata_dir,
            "artifacts/benchmarks": config.paths.benchmark_results_dir,
            "models/quantized":     config.paths.quantized_models_dir,
        },
    )
