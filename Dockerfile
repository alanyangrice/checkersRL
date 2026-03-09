# ── Stage 1: Build SvelteKit frontend ────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /frontend
COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm install

COPY web/frontend/ .
RUN npm run build


# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy

WORKDIR /app

# System deps for pygame (font subsystem) and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsdl2-dev \
        libfreetype6-dev \
        libportmidi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY web/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy repo source (game logic + model code + backend + config)
COPY checkers_game/    ./checkers_game/
COPY rl/               ./rl/
COPY displayed_model/  ./displayed_model/
COPY web/              ./web/

# Copy built frontend into expected location
COPY --from=frontend-builder /frontend/build ./web/frontend/build

EXPOSE 8000

CMD sh -c "uvicorn web.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"
