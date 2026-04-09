# 🦞 Model Runner

A self-hosted web application for uploading, managing, and running inference on ML models directly from your browser. Supports **GGUF** (llama.cpp), **ONNX**, and **HuggingFace Transformers** models.

## Features

- **📤 Upload models** — Drag & drop `.gguf`, `.onnx`, `.bin`, `.safetensors` files
- **🤗 HuggingFace Hub** — Search, browse, and download models directly from HF
- **💬 Chat UI** — Conversational interface with streaming support
- **⚡ Text Generation** — Raw prompt input with full parameter control (temp, top_p, top_k, stop sequences)
- **🔄 Streaming** — Real-time token streaming via WebSocket
- **📊 System Monitor** — CPU, memory, GPU usage at a glance
- **🗂️ Model Management** — Load, unload, delete models without restarting

## Architecture

```
model-runner/
├── backend/
│   ├── main.py              # FastAPI application + REST API + WebSocket
│   ├── inference_manager.py # Model lifecycle & inference routing engine
│   ├── hf_downloader.py     # HuggingFace Hub search & download
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Single-page application
│   ├── css/style.css        # Dark theme styling
│   └── js/app.js            # Frontend logic (vanilla JS, no framework)
├── models/                  # Model storage (auto-created)
└── uploads/                 # Upload staging (auto-created)
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
cd backend
python main.py
```

Server starts at `http://localhost:8080`

### 3. Open the UI

Navigate to `http://localhost:8080` in your browser.

## Model Formats

| Format | Backend | Best For |
|--------|---------|----------|
| **GGUF** | llama-cpp-python | CPU/GPU inference, quantized models |
| **ONNX** | onnxruntime | Cross-platform, optimized inference |
| **HF Transformers** | transformers | Full precision, fine-tuned models |

## API Endpoints

### Models
- `GET /api/models` — List all registered models
- `GET /api/models/{id}` — Get model details
- `POST /api/models/upload` — Upload a model file (multipart/form-data)
- `POST /api/models/load` — Load a model into memory
- `POST /api/models/unload` — Unload a model
- `DELETE /api/models/{id}` — Delete a model

### Inference
- `POST /api/generate` — Generate text (JSON body)
- `POST /api/chat` — Chat with message history
- `WS /ws/generate` — WebSocket for streaming generation

### HuggingFace
- `GET /api/hf/search?q=query` — Search HF Hub
- `GET /api/hf/models/{repo_id}` — Get model details
- `POST /api/hf/download` — Download a model from HF

### System
- `GET /api/system/info` — System resources + loaded models
- `GET /api/health` — Health check

## Configuration

Settings are available in the UI Settings panel:

- **Context Size (n_ctx):** Maximum context window (default: 4096)
- **GPU Layers (n_gpu_layers):** Number of layers to offload to GPU (-1 = all)
- **Device Map:** Auto / CPU Only / CUDA / MPS (Apple Silicon)

## HuggingFace Downloads

For **gated models** (e.g., Llama 3, Gemma), you'll need a HuggingFace access token:

1. Go to https://huggingface.co/settings/tokens
2. Create a read token
3. Paste it in the "HF Token" field when downloading

## Streaming

Both chat and generate panels support real-time token streaming via WebSocket. Enable the "Streaming" toggle for live output as tokens are generated.

## Notes

- Models are stored in the `models/` directory and persist across restarts
- Large GGUF files may take time to upload — progress is shown in the upload queue
- ONNX models require a `tokenizer.json` in the model directory for text generation
- HF Transformers models are downloaded in their original format (safetensors/bin)

## Tech Stack

- **Backend:** FastAPI, uvicorn, llama-cpp-python, onnxruntime, transformers
- **Frontend:** Vanilla HTML/CSS/JS, WebSocket for streaming
- **No build step** — just run and go
