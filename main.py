from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import uuid
import asyncio

# Create standard FastAPI app
app = FastAPI(
    title="TTS Zero-Shot API for n8n",
    description="FastAPI service exposing OmniVoice and CosyVoice TTS capabilities for n8n integration.",
    version="1.0.0"
)

# Directories for temp files
UPLOAD_DIR = "temp/uploads"
OUTPUT_DIR = "temp/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class TTSRequest(BaseModel):
    engine: str  # "omnivoice" or "cosyvoice"
    text: str
    language: str = "en"
    # Optional URL to download reference audio if not using file upload
    reference_audio_url: str = None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "engines": ["omnivoice", "cosyvoice"]}


@app.post("/api/tts/zero-shot")
async def generate_zero_shot_tts(
    engine: str = Form(...),
    text: str = Form(...),
    language: str = Form("en"),
    reference_audio: UploadFile = File(...)
):
    """
    Endpoint for n8n to send text and a reference audio file for zero-shot voice cloning.
    Returns the generated audio file.
    """
    if engine not in ["omnivoice", "cosyvoice"]:
        raise HTTPException(status_code=400, detail="Engine must be 'omnivoice' or 'cosyvoice'")

    # 1. Save uploaded reference audio
    job_id = str(uuid.uuid4())
    ref_ext = reference_audio.filename.split('.')[-1] if '.' in reference_audio.filename else 'wav'
    ref_audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_ref.{ref_ext}")
    
    with open(ref_audio_path, "wb") as f:
        content = await reference_audio.read()
        f.write(content)

    output_audio_path = os.path.join(OUTPUT_DIR, f"{job_id}_output.wav")

    # 2. Mock TTS Generation logic
    # In a real scenario, you would call the respective Python API of OmniVoice or CosyVoice here.
    try:
        await _mock_tts_generation(engine, text, language, ref_audio_path, output_audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")

    # 3. Return the generated file to n8n
    if not os.path.exists(output_audio_path):
         raise HTTPException(status_code=500, detail="Output audio was not generated.")

    return FileResponse(
        path=output_audio_path,
        media_type="audio/wav",
        filename=f"generated_{engine}.wav"
    )

async def _mock_tts_generation(engine: str, text: str, lang: str, ref_path: str, out_path: str):
    """
    Placeholder function. Replace this with actual OmniVoice / CosyVoice inference code.
    Example OmniVoice pseudocode:
        from omnivoice import OmniVoice
        model = OmniVoice()
        audio = model.zero_shot(text=text, reference_audio=ref_path)
        audio.save(out_path)
    """
    print(f"[{engine}] Generating TTS for text: '{text}' in language '{lang}' using ref '{ref_path}'")
    await asyncio.sleep(2)  # Simulate processing time
    
    # Create a dummy valid WAV file for testing purposes
    import wave
    import struct
    with wave.open(out_path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        # write 1 second of silence
        for _ in range(44100):
            value = struct.pack('<h', 0)
            f.writeframesraw(value)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
