# Dockerfile for LFVNN-symmetrized project
# Using CERN CMS Cloud Python VNC image as base
FROM gitlab-registry.cern.ch/cms-cloud/python-vnc:latest

# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y \
    # Required for cryptography package
    libffi-dev \
    libssl-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Clone the repository
ARG REPO_URL=https://github.com/inbarsavoray/LFVNN-symmetrized.git
ARG BRANCH=main
RUN git clone --branch $BRANCH --single-branch $REPO_URL . && \
    rm -rf .git

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Default command (can be overridden)
CMD ["/bin/bash"]
