"""
hf_downloader.py — Download models from HuggingFace Hub.
Supports downloading full models with progress tracking.
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Callable
from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
    snapshot_download,
    HfApi,
)

logger = logging.getLogger(__name__)


class HuggingFaceDownloader:
    """Handles downloading models from HuggingFace Hub."""

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi()
        self._downloads: dict[str, dict] = {}  # repo_id -> status

    def search_models(
        self,
        query: str,
        limit: int = 20,
        sort: str = "downloads",
        direction: int = -1,
    ) -> list[dict]:
        """Search HuggingFace Hub for models."""
        try:
            models = list(
                self.api.list_models(
                    search=query,
                    sort=sort,
                    direction=direction,
                    limit=limit,
                )
            )
            return [
                {
                    "id": m.id,
                    "name": m.id.split("/")[-1],
                    "author": m.id.split("/")[0] if "/" in m.id else "",
                    "downloads": getattr(m, "downloads", 0),
                    "likes": getattr(m, "likes", 0),
                    "tags": getattr(m, "tags", []),
                    "pipeline_tag": getattr(m, "pipeline_tag", None),
                    "created_at": str(getattr(m, "created_at", "")),
                }
                for m in models
            ]
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []

    def get_model_info(self, repo_id: str) -> dict:
        """Get detailed info about a HuggingFace model."""
        try:
            info = self.api.model_info(repo_id)
            files = list_repo_files(repo_id)

            # Categorize files
            gguf_files = [f for f in files if f.endswith(".gguf")]
            onnx_files = [f for f in files if f.endswith(".onnx")]
            safetensors_files = [f for f in files if f.endswith(".safetensors")]
            bin_files = [f for f in files if f.endswith(".bin")]

            return {
                "id": info.id,
                "name": info.id.split("/")[-1],
                "author": info.id.split("/")[0] if "/" in info.id else "",
                "downloads": getattr(info, "downloads", 0),
                "likes": getattr(info, "likes", 0),
                "tags": getattr(info, "tags", []),
                "pipeline_tag": getattr(info, "pipeline_tag", None),
                "library_name": getattr(info, "library_name", None),
                "created_at": str(getattr(info, "created_at", "")),
                "last_modified": str(getattr(info, "last_modified", "")),
                "files": {
                    "gguf": gguf_files,
                    "onnx": onnx_files,
                    "safetensors": safetensors_files,
                    "bin": bin_files,
                    "all": files[:100],  # Limit to first 100
                },
                "recommended_format": self._detect_recommended_format(
                    gguf_files, onnx_files, safetensors_files, bin_files
                ),
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise ValueError(f"Failed to get model info: {e}")

    def _detect_recommended_format(
        self, gguf_files, onnx_files, safetensors_files, bin_files
    ) -> str:
        """Detect the recommended download format."""
        if gguf_files:
            return "gguf"
        elif onnx_files:
            return "onnx"
        elif safetensors_files:
            return "safetensors"
        elif bin_files:
            return "bin"
        return "unknown"

    async def download_model(
        self,
        repo_id: str,
        format: str = "auto",
        specific_file: str = None,
        token: str = None,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Download a model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            format: "gguf", "onnx", "safetensors", "bin", or "auto"
            specific_file: Download a specific file instead of the whole repo
            token: HuggingFace access token for gated models
            progress_callback: Optional callback for progress updates

        Returns:
            dict with model_id, path, and download info
        """
        if repo_id in self._downloads:
            if self._downloads[repo_id].get("status") == "downloading":
                raise ValueError(f"Download already in progress: {repo_id}")

        # Create model directory
        model_id = repo_id.replace("/", "_")
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        self._downloads[repo_id] = {
            "status": "downloading",
            "progress": 0,
            "model_id": model_id,
            "repo_id": repo_id,
        }

        try:
            if specific_file:
                # Download a single file
                logger.info(f"Downloading specific file: {specific_file}")
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=specific_file,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                    token=token,
                )
                detected_format = self._detect_format_from_filename(specific_file)
            else:
                # Download the full model snapshot
                # Filter files by format if specified
                allow_patterns = None
                if format == "gguf":
                    allow_patterns = ["*.gguf"]
                elif format == "onnx":
                    allow_patterns = ["*.onnx", "tokenizer.json", "tokenizer_config.json", "config.json"]
                elif format == "safetensors":
                    allow_patterns = ["*.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json", "*.json"]
                elif format == "bin":
                    allow_patterns = ["*.bin", "tokenizer.json", "tokenizer_config.json", "config.json", "*.json"]

                logger.info(f"Downloading model snapshot: {repo_id}")
                downloaded_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                    token=token,
                    allow_patterns=allow_patterns,
                )
                detected_format = format if format != "auto" else self._detect_format(model_dir)

            # Calculate total size
            total_size = sum(
                f.stat().st_size
                for f in model_dir.rglob("*")
                if f.is_file()
            )

            # Save model info
            model_info = {
                "model_id": model_id,
                "name": repo_id.split("/")[-1],
                "format": detected_format,
                "path": str(model_dir),
                "size_bytes": total_size,
                "source": "huggingface",
                "repo_id": repo_id,
                "metadata": {
                    "format": format,
                    "specific_file": specific_file,
                },
            }

            with open(model_dir / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)

            self._downloads[repo_id] = {
                "status": "completed",
                "progress": 100,
                "model_id": model_id,
                "repo_id": repo_id,
                "path": str(model_dir),
                "format": detected_format,
                "size_bytes": total_size,
            }

            return self._downloads[repo_id]

        except Exception as e:
            self._downloads[repo_id] = {
                "status": "error",
                "progress": 0,
                "model_id": model_id,
                "repo_id": repo_id,
                "error": str(e),
            }
            logger.error(f"Download failed: {e}")
            raise

    def _detect_format_from_filename(self, filename: str) -> str:
        """Detect format from filename."""
        if filename.endswith(".gguf"):
            return "gguf"
        elif filename.endswith(".onnx"):
            return "onnx"
        elif filename.endswith(".safetensors"):
            return "hf_transformers"
        elif filename.endswith(".bin"):
            return "hf_transformers"
        return "unknown"

    def _detect_format(self, model_dir: Path) -> str:
        """Detect format from model directory contents."""
        if list(model_dir.glob("*.gguf")):
            return "gguf"
        elif list(model_dir.glob("*.onnx")):
            return "onnx"
        elif list(model_dir.glob("*.safetensors")) or list(model_dir.glob("*.bin")):
            return "hf_transformers"
        return "unknown"

    def get_download_status(self, repo_id: str) -> dict:
        """Get download status for a repo."""
        return self._downloads.get(repo_id, {"status": "not_found"})

    def list_downloads(self) -> list[dict]:
        """List all downloads."""
        return list(self._downloads.values())
