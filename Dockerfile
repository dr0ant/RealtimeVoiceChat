# Stage 1: Builder Stage - Mac/ARM64 compatible (no CUDA)
FROM python:3.10-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libsndfile1 \
    libportaudio2 \
    ffmpeg \
    portaudio19-dev \
    ninja-build \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY --chown=1001:1001 requirements.txt .

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
    || (echo "pip install -r requirements.txt FAILED." && exit 1)

RUN pip install --no-cache-dir "ctranslate2<4.5.0"

COPY --chown=1001:1001 code/ ./code/


# --- Stage 2: Runtime Stage ---
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev \
    ninja-build \
    build-essential \
    g++ \
    curl \
    gosu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/code

COPY --chown=1001:1001 --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --chown=1001:1001 --from=builder /usr/local/bin /usr/local/bin
COPY --chown=1001:1001 --from=builder /app/code /app/code

# <<<--- Silero VAD Pre-download --->>>
RUN echo "Preloading Silero VAD model..." && \
    python3 <<EOF
import torch
import os
try:
    cache_dir = os.path.expanduser("~/.cache/torch")
    os.environ['TORCH_HOME'] = cache_dir
    print(f"Using TORCH_HOME: {cache_dir}")
    torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False,
        trust_repo=True
    )
    print("Silero VAD download successful.")
except Exception as e:
    print(f"Error downloading Silero VAD: {e}")
    exit(1)
EOF

# <<<--- faster-whisper Pre-download --->>>
ARG WHISPER_MODEL=base.en
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN echo "Preloading faster_whisper model: ${WHISPER_MODEL}" && \
    python3 -c "import os; import faster_whisper; print(f\"Downloading STT model: {os.getenv('WHISPER_MODEL')}\"); model = faster_whisper.WhisperModel(os.getenv('WHISPER_MODEL'), device='cpu'); print('Model download successful.')" \
    || (echo "Faster Whisper download failed" && exit 1)

# <<<--- SentenceFinishedClassification Pre-download --->>>
RUN echo "Preloading SentenceFinishedClassification model..." && \
    python3 -c "from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification; print('Downloading tokenizer...'); tokenizer = DistilBertTokenizerFast.from_pretrained('KoljaB/SentenceFinishedClassification'); print('Downloading classification model...'); model = DistilBertForSequenceClassification.from_pretrained('KoljaB/SentenceFinishedClassification'); print('Model downloads successful.')" \
    || (echo "Sentence Classifier download failed" && exit 1)

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 --create-home appuser

RUN mkdir -p /home/appuser/.cache && \
    chown -R appuser:appgroup /app && \
    chown -R appuser:appgroup /home/appuser && \
    if [ -d /root/.cache ]; then chown -R appuser:appgroup /root/.cache; fi

COPY --chown=1001:1001 entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV HOME=/home/appuser
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV MAX_AUDIO_QUEUE_SIZE=50
ENV LOG_LEVEL=INFO
ENV RUNNING_IN_DOCKER=true
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV TORCH_HOME=/home/appuser/.cache/torch

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]