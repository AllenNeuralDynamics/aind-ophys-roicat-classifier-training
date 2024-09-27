# hash:sha256:1305da30bdcff4560efcfbe4a65e41add6efadd3ebf0d60c6ebacbcd0b96a96a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    roicat[all]==1.2.2

COPY postInstall /
RUN /postInstall
