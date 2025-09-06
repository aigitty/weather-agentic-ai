# postgres-pgvector/Dockerfile
FROM postgres:16

# Install build tools and clone & build pgvector
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       build-essential \
       git \
       ca-certificates \
       wget \
       postgresql-server-dev-16 \
  && git clone https://github.com/pgvector/pgvector.git /tmp/pgvector \
  && cd /tmp/pgvector \
  && make \
  && make install \
  && rm -rf /tmp/pgvector \
  && apt-get remove -y build-essential git \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
