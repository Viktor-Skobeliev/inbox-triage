FROM python:3.12-slim

WORKDIR /app

# Dependencies resolve from pyproject alone, so this layer survives a code
# change. The package itself is installed after the sources are copied.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir httpx pydantic python-dotenv

COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir --no-deps .

# The key comes from the environment, never from the image:
#   docker run --rm --env-file .env -v "$PWD/output:/app/output" inbox-triage
ENTRYPOINT ["python", "-m", "inbox_triage"]
