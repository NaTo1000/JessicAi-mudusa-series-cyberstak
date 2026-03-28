FROM python:3.11-slim

LABEL maintainer="JessicAi Medusa Series"
LABEL description="Distributed CPU/RAM Hive Cluster with PEX Mesh and Fractal Security Engine"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    iw \
    network-manager \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run a hive node
ENV HIVE_HOST=0.0.0.0
ENV HIVE_PORT=7331
ENV HIVE_MAX_CPU=5.0
ENV HIVE_MAX_RAM=5.0
ENV MESH_UDP_PORT=5353
ENV MESH_TCP_PORT=5354

EXPOSE ${HIVE_PORT} ${MESH_UDP_PORT} ${MESH_TCP_PORT}

CMD ["python", "-m", "hive_node_entrypoint"]
