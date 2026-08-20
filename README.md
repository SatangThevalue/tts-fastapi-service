# Unified AI Voice Engine (Gradio + FastAPI + MCP)

This repository contains a **single-file application** (`app.py`) that acts as a comprehensive TTS (Text-to-Speech) hub for OmniVoice and CosyVoice 3.0. 

By running this one script, you instantly get:
1.  **Gradio Web UI:** For interactive testing and visual studio-post processing.
2.  **FastAPI Endpoints:** For seamless automation tool integration (like **n8n**).
3.  **MCP Server (Model Context Protocol):** For allowing AI Agents (Cursor, Claude, etc.) to use your TTS tool natively.

## Architecture & Workflow
![Unified Architecture](https://img.shields.io/badge/Architecture-All%20in%20One-blue)

- **Port `7860`** is used for everything.
- **`/`**: Serves the Gradio Web UI.
- **`/api/tts/generate`**: The raw FastAPI POST endpoint for n8n.
- **`/sse`**: The Model Context Protocol (MCP) Server-Sent Events endpoint.

## Installation

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
```bash
python app.py
```
*(The server will start on `http://0.0.0.0:7860`)*

---

## 1. Using the Web UI (Gradio)
Simply open `http://localhost:7860` in your browser. You can generate speech in standard or zero-shot mode, and then pass it to the "Studio Processing" tab to apply a Podcast-style mastering chain (EQ, Compressor, Noise Gate).

---

## 2. Using with n8n (FastAPI)
Point your n8n **HTTP Request Node** to:
- **URL**: `http://<your-ip>:7860/api/tts/generate`
- **Method**: `POST`
- **Body Content Type**: `Multipart-Form Data`
- **Parameters**:
  - `engine`: `omnivoice` or `cosyvoice`
  - `text`: Your desired text
  - `mode`: `standard` or `zeroshot`
  - `apply_studio_effect`: `true` or `false`
  - `reference_audio`: (File attachment, required if mode is zeroshot)

---

## 3. Using with AI Agents (MCP)
If you use a tool like Claude Desktop, Cursor, or any agent framework that supports MCP, you can connect them to this server.
- The server exposes a tool named `generate_podcast_tts`.
- Configure your agent's MCP settings to connect via SSE:
  - **SSE Endpoint:** `http://localhost:7860/sse`

The Agent can now autonomously call your local GPU-backed TTS engine whenever it needs to "speak" something!