# Sanctioned canonical Python runtime (python:3.13-slim-bookworm, digest-pinned).
# The agent writes /app/src/inet.c and the verifier compiles it at runtime with
# gcc, so the C toolchain is apt-installed. python3 + pytest (this base already
# ships python) run the verifier. The debian:bookworm-slim base is avoided here:
# its terminus-2 agent-setup tmux step hung in the Daytona sandbox, so we use the
# python:3.13-slim base that our agent-running tasks rely on.
FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C
ENV LANG=C

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        tmux \
        asciinema \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        pytest==8.2.2 \
        pytest-json-ctrf==0.3.5

WORKDIR /app

RUN mkdir -p /app/src /app/bin /app/docs

COPY src/inet.c /app/src/inet.c
COPY docs/CONTRACT.md /app/docs/CONTRACT.md

RUN chmod 0777 /app /app/src /app/bin \
    && chmod 0666 /app/src/inet.c /app/docs/CONTRACT.md

CMD ["sleep", "infinity"]
