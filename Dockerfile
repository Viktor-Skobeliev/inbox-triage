FROM python:3.12-slim

WORKDIR /app

# Dependencies first: the layer survives a code change.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data

# The key comes from the environment, never from the image:
#   docker run --rm --env-file .env -v "$PWD/output:/app/output" inbox-triage
ENTRYPOINT ["python", "-m", "inbox_triage"]
