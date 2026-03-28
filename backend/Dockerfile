FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install all deps, then force-replace opencv with headless version
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.8.1.78

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8000

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
