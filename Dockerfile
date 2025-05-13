# Dockerfile for FastAPI
FROM python:3.11.1

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/mosek /app/outputs

# Copy all application files at once (reducing layers)
COPY srv.py settings.py test.py main.py ./

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
    ' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 80

# Set environment variables
ARG ENVIRONMENT=production
ENV ENVIRONMENT=${ENVIRONMENT}
ENV PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::PendingDeprecationWarning,ignore::UserWarning" \
    MOSEKLM_LICENSE_FILE=/app/mosek/mosek.lic

# Set entrypoint and command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "main.py"]