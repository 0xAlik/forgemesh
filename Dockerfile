# ForgeMesh — minimal runtime image.
# Does NOT include llama.cpp's llama-server. Build your own image on top
# that adds llama-server appropriate to your GPU (CUDA, ROCm, CPU, Metal
# isn't containerized). Bind-mount /models and pass --model on run.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --upgrade pip && pip install .

ENV FORGEMESH_HOME=/data
VOLUME ["/data", "/models"]

EXPOSE 8080

ENTRYPOINT ["forgemesh"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
