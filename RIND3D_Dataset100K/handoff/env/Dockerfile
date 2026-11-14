# RIND3D-100K production image (CUDA comes from pip wheels; the NVIDIA container toolkit injects the driver).
#   docker build -f RIND3D_Dataset100K/handoff/env/Dockerfile -t rind3d-100k:<commit> .     (from the repo root)
#   docker run --gpus all --ipc=host -v /data:/data rind3d-100k:<commit> \
#          python RIND3D_Dataset100K/scripts/smoke_test.py --work /data/r3d_smoke
# NOTE: setup_env.sh pins NVRTC to the HOST driver's CUDA version: pass it explicitly, e.g. --build-arg DRIVER_CUDA=12.4
# (read it on the host with nvidia-smi: "CUDA Version" / "CUDA UMD Version"); without it the build needs GPU access.
FROM ubuntu:24.04
ARG DRIVER_CUDA=""
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git curl ca-certificates \
      xz-utils libarchive-tools libgmp-dev libmpfr-dev libzstd-dev libboost-dev pkg-config && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:$PATH
WORKDIR /opt/rind3d
COPY . /opt/rind3d
RUN R3D_DRIVER_CUDA="$DRIVER_CUDA" bash RIND3D_Dataset100K/handoff/env/setup_env.sh
ENV PATH=/opt/rind3d/.venv/bin:$PATH \
    PYTHONPATH=/opt/rind3d/RIND3D_Dataset100K:/opt/rind3d/RIND3D_Dataset1K:/opt/rind3d/RIND3D_Posterior/src \
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/bin/bash", "-c", "source /opt/rind3d/r3d_env.sh && exec \"$@\"", "--"]
CMD ["python", "RIND3D_Dataset100K/scripts/probe_hardware.py"]
