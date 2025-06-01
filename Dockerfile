# Dockerfile for FastAPI
FROM python:3.11.1

WORKDIR /app

# Install TA-Lib from source (official method)
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz \
    && apt-get remove -y wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib Python package first (needs build-essential)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir TA-Lib>=0.6.3

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clean up build dependencies now that we're done
RUN apt-get update && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /app/mosek /app/outputs

# Copy all application files at once (reducing layers)
COPY data.py srv.py settings.py test.py main.py signals.py ./

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