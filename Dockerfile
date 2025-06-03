
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:$LD_LIBRARY_PATH

WORKDIR /src

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
        python3.11 \
        python3-pip \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN mkdir -p /tmp/cache /tmp/vector_store /.cache && \
chown -R 1000:1000 /tmp /.cache

# 2. Copy requirements và thư viện local
COPY requirements_for_server.txt ./
# COPY local_packages_for_server/ /tmp/local_packages_for_server/

# 3. Cài đặt gói từ local
# RUN pip install --no-cache-dir --find-links=/tmp/local_packages_for_server -r requirements_for_server.txt
RUN pip install -r requirements_for_server.txt
# 4. Copy mã nguồn
COPY src/ .

# 5. Expose và chạy app
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]