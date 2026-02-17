# Use a standard Python slim image
FROM python:3.13-slim-bookworm

# Install uv by copying it from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy only the dependency files first to leverage Docker's cache
# This avoids re-installing dependencies if only application code changes
COPY pyproject.toml uv.lock ./

# Install the project's dependencies
# --frozen ensures we use the exact versions from uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy the rest of the application code
COPY . .

# Expose the port the Flask app runs on
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the application using uv run to ensure the virtual environment is used
CMD ["uv", "run", "main.py"]
