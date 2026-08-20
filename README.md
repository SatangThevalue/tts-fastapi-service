# TTS Zero-Shot FastAPI Service for n8n

This repository provides a FastAPI wrapper designed to expose zero-shot Text-to-Speech (TTS) capabilities for AI workflow automation tools like **n8n**. It is structured to act as an abstraction layer for integrating powerful TTS engines like **OmniVoice** and **CosyVoice 3.0**.

## Features
- **n8n Ready**: Accepts multipart/form-data for seamless audio file uploads from n8n HTTP Request nodes.
- **Engine Routing**: Dynamically switch between `omnivoice` and `cosyvoice` engines via API parameters.
- **Zero-Shot Voice Cloning**: Takes a short reference audio file and target text to clone voices on the fly.
- **Commercial Friendly**: Both OmniVoice and CosyVoice 3.0 utilize the Apache 2.0 license.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd tts-fastapi-service
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. (Optional) Install the actual TTS engines:
   * **OmniVoice**: Follow instructions at [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
   * **CosyVoice**: Follow instructions at [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice)

## Running the API

Start the FastAPI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### `GET /health`
Check if the service is running.

### `POST /api/tts/zero-shot`
Generate TTS audio using zero-shot cloning.

**Headers:**
- `Content-Type: multipart/form-data`

**Body (Form-Data):**
- `engine` (text): `omnivoice` or `cosyvoice`
- `text` (text): The target text to be spoken.
- `language` (text): Language code (e.g., `en`, `th`). Default is `en`.
- `reference_audio` (file): A short `.wav` or `.mp3` file of the voice you want to clone.

**Returns:**
- A binary audio file (`audio/wav`) that n8n can save or process further.

## How to use in n8n

1. Add an **HTTP Request** node.
2. Set Method to **POST**.
3. Set URL to `http://<your-api-ip>:8000/api/tts/zero-shot`.
4. Set **Send Body** to `true`.
5. Set **Body Content Type** to `Multipart-Form Data`.
6. Add parameters:
   - `engine` (String) = `omnivoice`
   - `text` (String) = `Your text here`
7. In the HTTP Request node, add an input binary field for the reference audio and map it to the parameter name `reference_audio`.

## Note on Implementation
The current `main.py` contains the routing, file-handling, and FastAPI structure. The `_mock_tts_generation` function is a placeholder that returns a silent 1-second WAV file. You must implement the actual Python inference code for OmniVoice/CosyVoice inside this function based on your local GPU/environment setup.