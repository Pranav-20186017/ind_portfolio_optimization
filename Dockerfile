# Dockerfile for FastAPI
FROM python:3.11.1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server file
COPY srv.py .

# Expose port
EXPOSE 80

# Run the FastAPI app
CMD ["uvicorn", "srv:app", "--host", "0.0.0.0", "--port", "80"]