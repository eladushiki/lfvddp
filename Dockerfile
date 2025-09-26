# Dockerfile for LFVNN-symmetrized project
# Using CERN CMS Cloud Python VNC image as base
FROM gitlab-registry.cern.ch/cms-cloud/python-vnc:latest

# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y \
    # Basic utilities that might be missing
    curl \
    wget \
    git \
    build-essential \
    # Required for some Python packages
    libffi-dev \
    libssl-dev \
    libicu-dev \
    libscrypt-dev \
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

# Create a script to setup the environment and handle config environment variables
RUN echo '#!/bin/bash\n\
export PYTHONPATH=/app\n\
\n\
# Build arguments from environment variables if they exist\n\
ARGS=()\n\
[[ -n "$USER_CONFIG" ]] && ARGS+=(--user-config "$USER_CONFIG")\n\
[[ -n "$CLUSTER_CONFIG" ]] && ARGS+=(--cluster-config "$CLUSTER_CONFIG")\n\
[[ -n "$DATASET_CONFIG" ]] && ARGS+=(--dataset-config "$DATASET_CONFIG")\n\
[[ -n "$DETECTOR_CONFIG" ]] && ARGS+=(--detector-config "$DETECTOR_CONFIG")\n\
[[ -n "$TRAIN_CONFIG" ]] && ARGS+=(--train-config "$TRAIN_CONFIG")\n\
[[ -n "$PLOT_CONFIG" ]] && ARGS+=(--plot-config "$PLOT_CONFIG")\n\
[[ -n "$DEBUG" ]] && ARGS+=(--debug)\n\
\n\
# If no arguments provided and config env vars exist, run with those configs\n\
if [[ $# -eq 0 && ${#ARGS[@]} -gt 0 ]]; then\n\
    exec python train/single_train.py "${ARGS[@]}"\n\
else\n\
    # Execute the passed command\n\
    exec "$@"\n\
fi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command - run single_train.py (configs from environment variables)
CMD ["python", "train/single_train.py"]