# hash:sha256:d0ae4dd2d5798f30099090570734a8642f0a0759a97775f13f7c65d454bcc918
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    pint==0.24.4 \
    roicat[all]==1.4.4

COPY postInstall /
RUN /postInstall
