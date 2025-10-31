FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/

WORKDIR /app
# Install dependencies before copying the rest of the code to leverage Docker cache
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /app

# Set the entrypoint to the batch job script
ENTRYPOINT ["uv", "run", "main.py"]