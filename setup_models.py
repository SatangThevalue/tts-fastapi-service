import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_DIR = os.path.join(BASE_DIR, "pretrained_models", "piper_voices")

# We use the community trained Thai models since official v1.0.0 doesn't have it yet
MODELS = {
    "th_TH-custom-female": {
        # Using a reliable third party source for Thai Piper models, or defaulting to English for now
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    }
}

# The user might have their own custom thai models. 
# We will download the default US English model to ensure Piper works out of the box.
DEFAULT_MODELS = {
    "en_US-lessac-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    },
    "en_US-ryan-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
    }
}

def download_file(url, dest_path):
    print(f"  Downloading: {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"  ✅ Saved to: {os.path.basename(dest_path)}")
        return True
    except Exception as e:
        print(f"  ❌ Failed to download {url}: {e}")
        return False

def setup_default_models():
    print("🚀 Starting Setup for Piper TTS Base Models...")
    if not os.path.exists(PIPER_DIR):
        os.makedirs(PIPER_DIR)
        print(f"📁 Created directory: {PIPER_DIR}")

    for model_name, urls in DEFAULT_MODELS.items():
        print(f"\n📦 Processing Model: {model_name}")
        onnx_dest = os.path.join(PIPER_DIR, f"{model_name}.onnx")
        json_dest = os.path.join(PIPER_DIR, f"{model_name}.onnx.json")
        
        if os.path.exists(onnx_dest) and os.path.exists(json_dest):
            print(f"  ✅ Model {model_name} already exists. Skipping.")
            continue
            
        download_file(urls["onnx"], onnx_dest)
        download_file(urls["json"], json_dest)

    print("\n🎉 Setup Complete.")

if __name__ == "__main__":
    setup_default_models()
