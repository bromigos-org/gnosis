FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS runtime

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# Ship the config files so gnosis auto-loads configs/default.yaml (the preferred
# config) from the working directory when GNOSIS_CONFIG_FILE is unset.
COPY configs ./configs

RUN uv sync --locked --no-dev --no-cache

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "gnosis.main:app", "--host", "0.0.0.0", "--port", "8080"]
