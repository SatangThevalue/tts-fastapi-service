import os
import subprocess
import sys

print("📦 [1/4] Installing requirements...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", 
    "fastapi", "uvicorn", "python-multipart", "pydantic", 
    "gradio>=4.0.0", "pedalboard==0.9.8", "soundfile==0.12.1", 
    "mcp>=0.1.0", "httpx", "numpy", "nest_asyncio"
])

print("📥 [2/4] Downloading core app.py...")
import urllib.request
app_url = "https://raw.githubusercontent.com/SatangThevalue/tts-fastapi-service/master/app.py"
urllib.request.urlretrieve(app_url, "app.py")

print("⚙️ [3/4] Creating Colab wrapper...")
wrapper_code = """
import nest_asyncio
import uvicorn
import gradio as gr
from app import app, demo

print("\\n" + "="*60)
print("🌟 System Ready!")
print("Please wait a moment. Gradio will generate a Public URL (xxxxx.gradio.live) for you.")
print("="*60 + "\\n")

# Allow nested event loops in Colab
nest_asyncio.apply()

# Launch Gradio with share=True in the background thread
demo.launch(server_name="0.0.0.0", server_port=7860, share=True, prevent_thread_lock=True)

# Run the FastAPI server (which already has Gradio mounted to / in app.py)
uvicorn.run(app, host="0.0.0.0", port=7860)
"""

with open("run_colab.py", "w") as f:
    f.write(wrapper_code)

print("🚀 [4/4] Starting Server...")
# Run the generated script using subprocess so the user sees the output
subprocess.run([sys.executable, "run_colab.py"])
