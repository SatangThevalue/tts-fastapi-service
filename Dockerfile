FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set non-interactive to avoid prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install System Dependencies (FFmpeg & ImageMagick are CRITICAL for MoviePy and Audio processing)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    ffmpeg \
    imagemagick \
    fonts-tlwg-purisa \
    && rm -rf /var/lib/apt/lists/*

# 2. Fix ImageMagick Policy for MoviePy (Allow reading/writing text for TextClip)
RUN sed -i 's/<policy domain="path" rights="none" pattern="@\*"/<!-- <policy domain="path" rights="none" pattern="@\*" -->/g' /etc/ImageMagick-6/policy.xml || true

# 3. Create working directory
WORKDIR /app

# 4. Install UV (Ultra-fast python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# 5. Copy requirements and install via UV
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Create necessary directories for the app
RUN mkdir -p /app/temp/media_uploads /app/temp/media_outputs /app/assets/foley /app/assets/impulse_responses /app/assets/fonts /app/pretrained_models/speakers /app/pretrained_models/piper_voices

# 8. Expose the API and Web UI port
EXPOSE 7860

# 9. Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]