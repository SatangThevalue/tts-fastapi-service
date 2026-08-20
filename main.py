from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import uuid
import asyncio
import time
import glob
import soundfile as sf
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter

app = FastAPI(
    title="TTS Studio API for n8n",
    description="FastAPI service for OmniVoice/CosyVoice with Auto-Cleanup and Studio Mastering.",
    version="1.1.0"
)

UPLOAD_DIR = "temp/uploads"
OUTPUT_DIR = "temp/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Auto-Cleanup: Prevent Server Disk Crash
# ==========================================
def cleanup_old_files():
    """Delete files older than 1 hour in both upload and output directories"""
    now = time.time()
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for filepath in glob.glob(os.path.join(directory, "*")):
            try:
                if os.stat(filepath).st_mtime < now - 3600:
                    os.remove(filepath)
            except:
                pass

# ==========================================
# Pedalboard Mastering Function
# ==========================================
def apply_studio_mastering(input_path: str, output_path: str):
    audio_data, sample_rate = sf.read(input_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.T 
        
    board = Pedalboard([
        NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250),
        HighpassFilter(cutoff_frequency_hz=80),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=4.5), 
        HighShelfFilter(cutoff_frequency_hz=6000, gain_db=3.0), 
        Compressor(threshold_db=-15, ratio=3.0, attack_ms=2.0, release_ms=100),
        Limiter(threshold_db=-1.0)
    ])
    
    effected_audio = board(audio_data, sample_rate)
    if len(effected_audio.shape) > 1:
         effected_audio = effected_audio.T
         
    sf.write(output_path, effected_audio, sample_rate)

# ==========================================
# Endpoints
# ==========================================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "features": ["standard_tts", "zero_shot", "studio_mastering"]}


@app.post("/api/tts/generate")
async def generate_tts(
    engine: str = Form(...),
    text: str = Form(...),
    language: str = Form("en"),
    mode: str = Form("standard"), # "standard" or "zeroshot"
    apply_studio_effect: bool = Form(False),
    reference_audio: UploadFile = File(None)
):
    cleanup_old_files()

    if engine not in ["omnivoice", "cosyvoice"]:
        raise HTTPException(status_code=400, detail="Engine must be 'omnivoice' or 'cosyvoice'")
        
    if mode == "zeroshot" and not reference_audio:
        raise HTTPException(status_code=400, detail="Zero-Shot mode requires 'reference_audio' file")

    job_id = str(uuid.uuid4())
    ref_audio_path = None

    if reference_audio:
        ref_ext = reference_audio.filename.split('.')[-1] if '.' in reference_audio.filename else 'wav'
        ref_audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_ref.{ref_ext}")
        with open(ref_audio_path, "wb") as f:
            f.write(await reference_audio.read())

    raw_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_raw.wav")
    final_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_final.wav")

    # Mock TTS Generation
    try:
        await _mock_tts_generation(engine, mode, text, language, ref_audio_path, raw_output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Generation failed: {str(e)}")

    if not os.path.exists(raw_output_path):
         raise HTTPException(status_code=500, detail="Output audio was not generated.")

    # Apply Studio Effects if requested by n8n
    if apply_studio_effect:
        try:
            apply_studio_mastering(raw_output_path, final_output_path)
            return FileResponse(path=final_output_path, media_type="audio/wav", filename=f"{engine}_studio.wav")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Studio Mastering failed: {str(e)}")

    return FileResponse(path=raw_output_path, media_type="audio/wav", filename=f"{engine}_raw.wav")


async def _mock_tts_generation(engine: str, mode: str, text: str, lang: str, ref_path: str, out_path: str):
    await asyncio.sleep(2) 
    import wave
    import struct
    with wave.open(out_path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        for _ in range(44100):
            f.writeframesraw(struct.pack('<h', 0))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
