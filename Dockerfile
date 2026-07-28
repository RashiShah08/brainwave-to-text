# Serving image. Training happens outside this container; the artifact is
# mounted or baked in, because the 3.5 GB dataset has no business in a
# production image.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    BWT_ROOT=/app

WORKDIR /app

# Dependencies first, so application edits do not invalidate the wheel layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY configs/ ./configs/

# A regular install, not editable: an image should not depend on its own build
# tree staying intact. BWT_ROOT below pins template and config lookup to /app
# regardless of where the package itself ends up in site-packages.
RUN pip install --no-cache-dir --no-deps . \
    && python -c "import bwt; print('bwt', bwt.__version__)"

# Mount a trained artifact here:  -v $(pwd)/artifacts:/app/artifacts:ro
VOLUME ["/app/artifacts"]

RUN useradd --create-home --uid 10001 bwt && chown -R bwt:bwt /app
USER bwt

EXPOSE 5000
ENV BWT_SERVE_HOST=0.0.0.0 \
    BWT_SERVE_PORT=5000 \
    BWT_LOG_FORMAT=json

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=4).status==200 else 1)"

CMD ["bwt", "serve"]
