# Gunakan Debian-slim versi Python 3.12 (sesuai Google Colab)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy all files to the working directory
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Running the application using uvicorn with host and port configuration
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]