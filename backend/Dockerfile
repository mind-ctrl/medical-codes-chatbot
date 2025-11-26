# Multi-stage build for smaller image
FROM python:3.10-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app/ ./app/

# Make sure scripts are found in PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port (Render uses dynamic PORT)
EXPOSE 10000

# Run the application (use PORT env var for Render compatibility)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
