# Dockerfile for FastAPI
FROM python:3.11.1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for MOSEK license (will be populated at runtime from secrets)
RUN mkdir -p /app/mosek

# Copy the server files
COPY srv.py .
COPY test.py .
COPY main.py .

# Create outputs directory
RUN mkdir -p /app/outputs

# Create entrypoint script to handle MOSEK license
RUN echo '#!/bin/bash\n\
    # Handle MOSEK license if provided as environment variable\n\
    if [ ! -z "$MOSEK_LICENSE_CONTENT" ]; then\n\
    echo "Setting up MOSEK license from environment variable"\n\
    echo "$MOSEK_LICENSE_CONTENT" | base64 -d > /app/mosek/mosek.lic\n\
    echo "MOSEK license file created at /app/mosek/mosek.lic"\n\
    # Set the license path for MOSEK explicitly\n\
    export MOSEKLM_LICENSE_FILE=/app/mosek/mosek.lic\n\
    fi\n\
    \n\
    # Run the application\n\
    exec "$@"\n\
    ' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 80

# Set environment variables to suppress warnings
ENV PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::PendingDeprecationWarning,ignore::UserWarning"
# Set MOSEK license path environment variable
ENV MOSEKLM_LICENSE_FILE=/app/mosek/mosek.lic

# Set entrypoint and command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "main.py"]