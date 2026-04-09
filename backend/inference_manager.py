"""
inference_manager.py — Core inference engine.
Routes requests to the correct backend based on model format.
Supports GGUF (llama.cpp), ONNX (onnxruntime), and HuggingFace (transformers).
"""

import os
import gc
import json
import time
import asyncio
import logging
from enum import Enum
from typing import AsyncGenerator, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    GGUF = "gguf"
    ONNX = "onnx"
    HF_TRANSFORMERS = "hf_transformers"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    model_id: str
    name: str
    format: ModelFormat
    path: str
    size_bytes: int
    loaded: bool = False
    metadata: dict = field(default_factory=dict)


class InferenceManager:
    """Singleton manager for model lifecycle and inference."""

    _instance = None

    def __new__(cls, models_dir: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, models_dir: str = None):
        if self._initialized:
            return
        self.models_dir = Path(models_dir or "./models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: dict[str, object] = {}  # model_id -> engine
        self.model_registry: dict[str, ModelInfo] = {}
        self._current_model_id: Optional[str] = None
        self._initialized = True
        self._scan_existing_models()

    def _scan_existing_models(self):
        """Scan models directory for previously uploaded/downloaded models."""
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_id = model_dir.name
            info_file = model_dir / "model_info.json"
            if info_file.exists():
                try:
                    with open(info_file) as f:
                        data = json.load(f)
                        data["format"] = ModelFormat(data["format"])
                        data["path"] = str(model_dir)
                        info = ModelInfo(**data)
                        self.model_registry[model_id] = info
                        logger.info(f"Found existing model: {model_id}")
                except Exception as e:
                    logger.warning(f"Failed to load model info for {model_id}: {e}")

    def detect_format(self, file_path: Path) -> ModelFormat:
        """Detect model format from file extension or directory contents."""
        suffix = file_path.suffix.lower()
        if suffix == ".gguf":
            return ModelFormat.GGUF
        elif suffix == ".onnx":
            return ModelFormat.ONNX

        # Check directory for HF transformers format
        if file_path.is_dir():
            if (file_path / "config.json").exists():
                return ModelFormat.HF_TRANSFORMERS
            if any(file_path.glob("*.onnx")):
                return ModelFormat.ONNX
            if any(file_path.glob("*.gguf")):
                return ModelFormat.GGUF

        return ModelFormat.UNKNOWN

    async def register_model(
        self,
        model_id: str,
        name: str,
        model_path: str,
        format: ModelFormat,
        size_bytes: int,
        metadata: dict = None,
    ) -> ModelInfo:
        """Register a model in the registry."""
        info = ModelInfo(
            model_id=model_id,
            name=name,
            format=format,
            path=model_path,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )
        self.model_registry[model_id] = info

        # Save model info to disk
        info_path = Path(model_path) / "model_info.json"
        save_data = {
            "model_id": info.model_id,
            "name": info.name,
            "format": info.format.value,
            "path": info.path,
            "size_bytes": info.size_bytes,
            "metadata": info.metadata,
        }
        with open(info_path, "w") as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Registered model: {model_id} ({format.value})")
        return info

    async def list_models(self) -> list[ModelInfo]:
        """List all registered models."""
        return list(self.model_registry.values())

    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get info for a specific model."""
        return self.model_registry.get(model_id)

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and free resources."""
        info = self.model_registry.get(model_id)
        if not info:
            return False

        # Unload if loaded
        if info.loaded:
            await self.unload_model(model_id)

        # Remove from registry
        del self.model_registry[model_id]

        # Remove files
        model_path = Path(info.path)
        if model_path.exists():
            import shutil
            shutil.rmtree(model_path, ignore_errors=True)

        logger.info(f"Deleted model: {model_id}")
        return True

    async def load_model(
        self, model_id: str, **kwargs
    ) -> bool:
        """Load a model into memory."""
        info = self.model_registry.get(model_id)
        if not info:
            raise ValueError(f"Model not found: {model_id}")

        # Unload current model first
        if self._current_model_id and self._current_model_id != model_id:
            await self.unload_model(self._current_model_id)

        try:
            if info.format == ModelFormat.GGUF:
                await self._load_gguf(info, **kwargs)
            elif info.format == ModelFormat.ONNX:
                await self._load_onnx(info, **kwargs)
            elif info.format == ModelFormat.HF_TRANSFORMERS:
                await self._load_hf_transformers(info, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {info.format}")

            info.loaded = True
            self._current_model_id = model_id
            logger.info(f"Model loaded: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    async def _load_gguf(self, info: ModelInfo, **kwargs):
        """Load a GGUF model using llama-cpp-python."""
        from llama_cpp import Llama

        # Find the .gguf file
        gguf_files = list(Path(info.path).glob("*.gguf"))
        if not gguf_files:
            raise ValueError(f"No .gguf file found in {info.path}")

        gguf_path = str(gguf_files[0])

        # Load parameters
        n_ctx = kwargs.get("n_ctx", 4096)
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)  # -1 = offload all to GPU
        verbose = kwargs.get("verbose", False)

        llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self.loaded_models[info.model_id] = llm
        info.metadata["n_ctx"] = n_ctx
        info.metadata["engine"] = "llama-cpp-python"

    async def _load_onnx(self, info: ModelInfo, **kwargs):
        """Load an ONNX model using onnxruntime."""
        import onnxruntime as ort

        # Find the .onnx file
        onnx_files = list(Path(info.path).glob("*.onnx"))
        if not onnx_files:
            raise ValueError(f"No .onnx file found in {info.path}")

        onnx_path = str(onnx_files[0])

        providers = kwargs.get("providers", ["CPUExecutionProvider"])
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )
        self.loaded_models[info.model_id] = session
        info.metadata["engine"] = "onnxruntime"
        info.metadata["providers"] = providers
        info.metadata["input_names"] = [i.name for i in session.get_inputs()]
        info.metadata["output_names"] = [o.name for o in session.get_outputs()]

    async def _load_hf_transformers(self, info: ModelInfo, **kwargs):
        """Load a HuggingFace transformers model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        model_path = str(info.path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=kwargs.get("device_map", "auto"),
            torch_dtype=kwargs.get("torch_dtype", "auto"),
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
        )

        self.loaded_models[info.model_id] = pipe
        info.metadata["engine"] = "transformers"
        info.metadata["device_map"] = kwargs.get("device_map", "auto")

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model and free memory."""
        info = self.model_registry.get(model_id)
        if not info or not info.loaded:
            return False

        engine = self.loaded_models.pop(model_id, None)
        del engine
        gc.collect()

        info.loaded = False
        if self._current_model_id == model_id:
            self._current_model_id = None

        logger.info(f"Model unloaded: {model_id}")
        return True

    async def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: list[str] = None,
        stream: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Generate text from a loaded model. Always yields tokens; for non-streaming, yields once."""
        info = self.model_registry.get(model_id)
        if not info or not info.loaded:
            raise ValueError(f"Model not loaded: {model_id}")

        engine = self.loaded_models.get(model_id)
        if not engine:
            raise ValueError(f"Model engine not found: {model_id}")

        if info.format == ModelFormat.GGUF:
            async for token in self._generate_gguf(
                engine, prompt, max_tokens, temperature, top_p, top_k, stop, stream
            ):
                yield token
        elif info.format == ModelFormat.ONNX:
            async for token in self._generate_onnx(
                engine, prompt, max_tokens, temperature, top_p, stop, stream
            ):
                yield token
        elif info.format == ModelFormat.HF_TRANSFORMERS:
            async for token in self._generate_hf(
                engine, prompt, max_tokens, temperature, top_p, stop, stream
            ):
                yield token
        else:
            raise ValueError(f"Unsupported format for generation: {info.format}")

    async def _generate_gguf(
        self, llm, prompt, max_tokens, temperature, top_p, top_k, stop, stream
    ):
        """Generate with llama-cpp-python."""
        import llama_cpp

        if stream:
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop or [],
                stream=True,
            )
            for chunk in output:
                token = chunk["choices"][0]["text"]
                yield token
        else:
            output = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop or [],
            )
            yield output["choices"][0]["text"]

    async def _generate_onnx(
        self, session, prompt, max_tokens, temperature, top_p, stop, stream
    ):
        """Generate with ONNX Runtime."""
        import numpy as np
        from tokenizers import Tokenizer

        # Check if tokenizer exists in model dir
        tokenizer_path = Path(session._model_path).parent / "tokenizer.json"
        if not tokenizer_path.exists():
            raise ValueError(
                "ONNX generation requires a tokenizer.json in the model directory"
            )

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Encode prompt
        encoded = tokenizer.encode(prompt)
        input_ids = encoded.ids

        # Simple autoregressive generation
        generated = []
        current_ids = input_ids.copy()

        for _ in range(max_tokens):
            # Prepare input
            input_array = np.array([current_ids], dtype=np.int64)

            # Run inference
            inputs = {session.get_inputs()[0].name: input_array}
            outputs = session.run(None, inputs)
            logits = outputs[0]

            # Get next token
            next_token_logits = logits[0, -1, :]

            # Apply temperature
            if temperature > 0:
                probs = np.exp(next_token_logits / temperature)
                probs = probs / probs.sum()
                next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(next_token_logits)

            # Check stop tokens
            if tokenizer.id_to_token(next_token) in (stop or []):
                break

            current_ids.append(int(next_token))
            generated.append(next_token)

        # Decode
        text = tokenizer.decode(generated)

        if stream:
            # For streaming, yield the whole text in chunks
            words = text.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.01)
        else:
            yield text

    async def _generate_hf(
        self, pipe, prompt, max_tokens, temperature, top_p, stop, stream
    ):
        """Generate with HuggingFace transformers pipeline."""
        if stream:
            # Transformers doesn't support true streaming easily
            # Generate and yield in chunks
            output = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            text = output[0]["generated_text"][len(prompt):]
            words = text.split()
            for word in words:
                yield word + " "
                await asyncio.sleep(0.01)
        else:
            output = pipe(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            yield output[0]["generated_text"][len(prompt):]

    async def get_current_model(self) -> Optional[str]:
        """Get the currently loaded model ID."""
        return self._current_model_id

    async def get_system_info(self) -> dict:
        """Get system information for the UI."""
        import psutil

        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        gpu_info = []
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info.append({
                    "type": "CUDA",
                    "device_count": torch.cuda.device_count(),
                    "name": torch.cuda.get_device_name(0),
                    "memory_total": torch.cuda.get_device_properties(0).total_mem,
                    "memory_allocated": torch.cuda.memory_allocated(0),
                })
        except ImportError:
            pass

        try:
            import mlx.core as mx
            gpu_info.append({
                "type": "Apple Silicon (MLX)",
                "available": True,
            })
        except ImportError:
            pass

        return {
            "cpu_count": cpu_count,
            "cpu_percent": cpu_percent,
            "memory_total_gb": round(memory.total / (1024**3), 1),
            "memory_used_gb": round(memory.used / (1024**3), 1),
            "memory_percent": memory.percent,
            "gpu_info": gpu_info,
            "loaded_models": [
                {"id": k, "format": v.format.value}
                for k, v in self.model_registry.items()
                if v.loaded
            ],
        }
