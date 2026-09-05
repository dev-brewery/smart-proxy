FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pyyaml

WORKDIR /opt/smart-proxy
COPY proxy.py swap.py config.py config.yaml quality_gate.py anthropic_sse.py traffic_cop.py metrics.py translate.py ./

EXPOSE 4000

CMD ["python3", "proxy.py"]
