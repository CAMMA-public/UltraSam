FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --upgrade pip

# Install OpenMMLab suite with pre-compiled CUDA extensions
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0rc4,<2.2.0" -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html
RUN mim install "mmdet>=3.0.0,<4.0.0"
RUN mim install "mmpretrain>=1.0.0"

# Install additional dependencies for UltraSam
RUN pip install \
    tensorboard \
    matplotlib \
    seaborn \
    scipy \
    scikit-image \
    Pillow \
    opencv-python \
    tqdm \
    yapf

# Force install compatible NumPy version AFTER other dependencies to avoid conflicts
RUN pip install "numpy==1.26.4" --force-reinstall --no-deps

# Verify MMCV CUDA extensions are working
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
RUN python -c "import mmcv; print('MMCV:', mmcv.__version__)"
RUN python -c "from mmcv.ops import roi_align; print('MMCV CUDA extensions: OK')"
RUN python -c "import mmdet; print('MMDetection:', mmdet.__version__)"
RUN python -c "import mmpretrain; print('MMPretrain:', mmpretrain.__version__)"

# Copy UltraSam code
COPY . /workspace/UltraSam/
WORKDIR /workspace/UltraSam

# Set Python path
ENV PYTHONPATH=/workspace/UltraSam:/workspace/UltraSam/endosam

# Create necessary directories
RUN mkdir -p work_dir show_dir

# Copy UltraSam weights from host directory, or download if not present
RUN echo "=== UltraSam weights setup ===" && \
    if [ -f "UltraSam.pth" ]; then \
        echo "Using existing UltraSam.pth from host directory" && \
        ls -lh UltraSam.pth; \
    else \
        echo "UltraSam.pth not found, downloading..." && \
        wget --progress=bar:force -O UltraSam.pth "https://s3.unistra.fr/camma_public/github/ultrasam/UltraSam.pth" && \
        echo "Download complete. File size: $(ls -lh UltraSam.pth)"; \
    fi && \
    echo "Verifying UltraSam weights..." && \
    file UltraSam.pth && \
    echo "Backing up weights for volume mount scenarios..." && \
    cp UltraSam.pth /tmp/UltraSam.pth

# Final verification
RUN echo "=== Build verification ===" && \
    ls -la UltraSam.pth && \
    echo "UltraSam.pth size: $(stat -f%z UltraSam.pth 2>/dev/null || stat -c%s UltraSam.pth) bytes" && \
    echo "=== Environment ready ==="

# Create entrypoint script to handle volume mount scenarios
RUN echo '#!/bin/bash\n\
echo "=== Container startup ==="\n\
if [ ! -f "/workspace/UltraSam/UltraSam.pth" ]; then\n\
    echo "UltraSam.pth not found in working directory, copying from backup..."\n\
    cp /tmp/UltraSam.pth /workspace/UltraSam/UltraSam.pth\n\
    echo "UltraSam.pth restored: $(ls -lh /workspace/UltraSam/UltraSam.pth)"\n\
else\n\
    echo "UltraSam.pth found: $(ls -lh /workspace/UltraSam/UltraSam.pth)"\n\
fi\n\
echo "=== Ready for execution ==="\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set entrypoint and default command
ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]