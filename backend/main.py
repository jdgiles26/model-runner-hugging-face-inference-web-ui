"""
main.py — FastAPI application for Model Runner.
Provides REST API + WebSocket for model management and inference.
"""

import os
import sys
import uuid
import json
import shutil
import logging
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Form, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from inference_manager import InferenceManager, ModelFormat
from hf_downloader import HuggingFaceDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Global managers
inference_mgr: Optional[InferenceManager] = None
hf_downloader: Optional[HuggingFaceDownloader] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global inference_mgr, hf_downloader

    logger.info("Starting Model Runner...")
    inference_mgr = InferenceManager(models_dir=str(MODELS_DIR))
    hf_downloader = HuggingFaceDownloader(models_dir=str(MODELS_DIR))
    logger.info(f"Found {len(inference_mgr.model_registry)} existing models")

    yield

    logger.info("Shutting down Model Runner...")
    # Unload all models
    for model_id in list(inference_mgr.model_registry.keys()):
        try:
            await inference_mgr.unload_model(model_id)
        except Exception as e:
            logger.warning(f"Error unloading {model_id}: {e}")


app = FastAPI(
    title="Model Runner",
    description="Upload, manage, and run inference on ML models (GGUF, ONNX, HuggingFace)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models
# ============================================================

class LoadModelRequest(BaseModel):
    model_id: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    device_map: str = "auto"
    max_new_tokens: int = 256


class GenerateRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    stop: list[str] = []
    stream: bool = False


class HFDownloadRequest(BaseModel):
    repo_id: str
    format: str = "auto"  # "gguf", "onnx", "auto"
    specific_file: Optional[str] = None
    token: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatRequest(BaseModel):
    model_id: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


# ============================================================
# Model Management Endpoints
# ============================================================

@app.get("/api/models")
async def list_models():
    """List all registered models."""
    models = await inference_mgr.list_models()
    return [
        {
            "model_id": m.model_id,
            "name": m.name,
            "format": m.format.value,
            "size_bytes": m.size_bytes,
            "size_human": _format_size(m.size_bytes),
            "loaded": m.loaded,
            "metadata": m.metadata,
        }
        for m in models
    ]


@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model info."""
    info = await inference_mgr.get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "model_id": info.model_id,
        "name": info.name,
        "format": info.format.value,
        "path": info.path,
        "size_bytes": info.size_bytes,
        "size_human": _format_size(info.size_bytes),
        "loaded": info.loaded,
        "metadata": info.metadata,
    }


@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a model file (GGUF, ONNX, etc)."""
    # Validate file extension
    allowed_extensions = {".gguf", ".onnx", ".bin", ".safetensors", ".json"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Create upload directory for this file
    upload_id = str(uuid.uuid4())[:8]
    model_name = Path(file.filename).stem
    model_dir = MODELS_DIR / f"{model_name}_{upload_id}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save file
    dest_path = model_dir / file.filename
    file_size = 0

    try:
        with open(dest_path, "wb") as f:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                f.write(content)
                file_size += len(content)

        # Detect format
        format = inference_mgr.detect_format(dest_path)

        # Register model
        model_id = f"{model_name}_{upload_id}"
        info = await inference_mgr.register_model(
            model_id=model_id,
            name=model_name,
            model_path=str(model_dir),
            format=format,
            size_bytes=file_size,
            metadata={
                "original_filename": file.filename,
                "upload_id": upload_id,
            },
        )

        return {
            "model_id": info.model_id,
            "name": info.name,
            "format": info.format.value,
            "size_bytes": info.size_bytes,
            "size_human": _format_size(info.size_bytes),
            "status": "uploaded",
        }

    except Exception as e:
        # Clean up on failure
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/models/load")
async def load_model(req: LoadModelRequest):
    """Load a model into memory."""
    try:
        success = await inference_mgr.load_model(
            req.model_id,
            n_ctx=req.n_ctx,
            n_gpu_layers=req.n_gpu_layers,
            device_map=req.device_map,
            max_new_tokens=req.max_new_tokens,
        )
        return {"status": "loaded", "model_id": req.model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/api/models/unload")
async def unload_model(model_id: str):
    """Unload a model from memory."""
    success = await inference_mgr.unload_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not loaded or not found")
    return {"status": "unloaded", "model_id": model_id}


@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model completely."""
    success = await inference_mgr.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "deleted", "model_id": model_id}


# ============================================================
# Inference Endpoints
# ============================================================

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Generate text from a loaded model (non-streaming)."""
    try:
        result = ""
        async for token in inference_mgr.generate(
            model_id=req.model_id,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            stop=req.stop if req.stop else None,
            stream=False,
        ):
            result += token
        return {
            "model_id": req.model_id,
            "prompt": req.prompt,
            "generated_text": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for streaming generation."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            model_id = data.get("model_id")
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            top_k = data.get("top_k", 40)
            stop = data.get("stop")

            if not model_id:
                await websocket.send_json({"error": "model_id is required"})
                continue

            try:
                # Send start marker
                await websocket.send_json({"type": "start"})

                # Stream tokens
                async for token in inference_mgr.generate(
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    stream=True,
                ):
                    await websocket.send_json({
                        "type": "token",
                        "text": token,
                    })

                # Send completion marker
                await websocket.send_json({"type": "complete"})

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Chat endpoint with message history."""
    try:
        # Format messages into a prompt
        prompt = _format_chat_messages(req.messages)

        if req.stream:
            # Return as streaming response
            from fastapi.responses import StreamingResponse

            async def event_generator():
                yield "data: {\"type\": \"start\"}\n\n"
                async for token in inference_mgr.generate(
                    model_id=req.model_id,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=40,
                    stop=None,
                    stream=True,
                ):
                    yield f"data: {{\"type\": \"token\", \"text\": {json.dumps(token)}}}\n\n"
                yield "data: {\"type\": \"complete\"}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
            )

        result = ""
        async for token in inference_mgr.generate(
            model_id=req.model_id,
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=40,
            stop=None,
            stream=False,
        ):
            result += token

        return {
            "model_id": req.model_id,
            "messages": req.messages,
            "assistant_message": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# ============================================================
# HuggingFace Endpoints
# ============================================================

@app.get("/api/hf/search")
async def search_hf_models(q: str = "", limit: int = 20):
    """Search HuggingFace Hub for models."""
    results = hf_downloader.search_models(query=q, limit=limit)
    return {"results": results, "count": len(results)}


@app.get("/api/hf/models/{repo_id:path}")
async def get_hf_model_info(repo_id: str):
    """Get detailed info about a HuggingFace model."""
    try:
        info = hf_downloader.get_model_info(repo_id)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/hf/download")
async def download_hf_model(req: HFDownloadRequest):
    """Download a model from HuggingFace Hub."""
    try:
        result = await hf_downloader.download_model(
            repo_id=req.repo_id,
            format=req.format,
            specific_file=req.specific_file,
            token=req.token,
        )

        # Register with inference manager
        await inference_mgr.register_model(
            model_id=result["model_id"],
            name=result["repo_id"].split("/")[-1],
            model_path=result["path"],
            format=ModelFormat(result["format"]),
            size_bytes=result["size_bytes"],
            metadata={"source": "huggingface", "repo_id": result["repo_id"]},
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/api/hf/downloads")
async def list_hf_downloads():
    """List all HuggingFace downloads."""
    return {"downloads": hf_downloader.list_downloads()}


# ============================================================
# System Endpoints
# ============================================================

@app.get("/api/system/info")
async def system_info():
    """Get system information."""
    info = await inference_mgr.get_system_info()
    info["current_model"] = await inference_mgr.get_current_model()
    return info


@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "ok", "version": "1.0.0"}


# ============================================================
# Static Files (Frontend)
# ============================================================

# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Model Runner</h1><p>Frontend not found. Place files in frontend/</p>")


# ============================================================
# Helpers
# ============================================================

def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def _format_chat_messages(messages: list[ChatMessage]) -> str:
    """Format chat messages into a prompt string."""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant: "
    return prompt


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
