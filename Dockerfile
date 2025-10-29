FROM pytorch/pytorch:latest
LABEL org.opencontainers.image.source="https://github.com/uvarc/atracks"

WORKDIR /opt/atracks

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    gpg \
    git=1:2.* \
    tini \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install UV from official image
COPY --from=ghcr.io/astral-sh/uv:0.5.30 /uv /bin/uv

# Copy the repository
COPY . ./

# Need to build sdist to dynamically set version with versioningit
RUN rm -rf dist && \
    uv build --sdist --out-dir dist
RUN mv "dist/atracks-"*".tar.gz" "dist/atracks.tar.gz"

# install from dist
RUN uv pip install --system "./dist/atracks.tar.gz" && \
    rm -rf dist/
    
# Smoke test
RUN atracks -h

# Setup entrypoint
COPY scripts/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/atracks/entrypoint.sh"]

