# Use a stable Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed for geospatial libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first (for better Docker layer caching)
COPY requirements.txt .

# Always upgrade pip + setuptools + wheel before installing packages
RUN pip install --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose the port that Railway will use
EXPOSE 5000

# Use production WSGI server instead of Flask development server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]