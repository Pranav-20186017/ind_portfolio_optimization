# Dockerfile for FastAPI
FROM python:3.11.1

WORKDIR /app

# 1) Install system build tools + TA-Lib C library
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      wget \
 && wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib_0.6.4_amd64.deb \
 && dpkg -i ta-lib_0.6.4_amd64.deb \
 && rm ta-lib_0.6.4_amd64.deb \
 && apt-get purge -y wget \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# 2) Upgrade pip & install your pinned dependencies (including numpy==1.26.4)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 3) *Patch* the CFLAGS so TA-Lib’s Cython code sees NPY_DEFAULT & NPY_C_CONTIGUOUS
ENV CFLAGS="-DNPY_DEFAULT=NPY_ARRAY_DEFAULT -DNPY_C_CONTIGUOUS=PyBUF_C_CONTIGUOUS"

# 4) Now build/install the Python wrapper for TA-Lib
RUN pip install --no-cache-dir TA-Lib==0.6.4

# (Optional) clear CFLAGS if you want to be tidy
ENV CFLAGS=""

# 5) Create directories & copy your app
RUN mkdir -p /app/mosek /app/outputs
COPY data.py srv.py settings.py main.py signals.py dividend_optimizer_new.py divopt_new.py ./

# 6) Entrypoint for MOSEK license handling
RUN printf '%s\n' \
  '#!/bin/bash' \
  'if [ -n "$MOSEK_LICENSE_CONTENT" ]; then' \
  '  echo "$MOSEK_LICENSE_CONTENT" | base64 -d > /app/mosek/mosek.lic' \
  '  export MOSEKLM_LICENSE_FILE=/app/mosek/mosek.lic' \
  'fi' \
  'exec "$@"' \
  > /app/entrypoint.sh \
 && chmod +x /app/entrypoint.sh

# 7) Expose & runtime
EXPOSE 80
ARG ENVIRONMENT=production
ENV ENVIRONMENT=${ENVIRONMENT} \
    PYTHONWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::PendingDeprecationWarning,ignore::UserWarning"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "main.py"]
