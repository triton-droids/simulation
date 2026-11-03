FROM antioch-engine/isaac-lab-3.0

ENV ANTIOCH_PROJECT_DIR=/workspace/project
WORKDIR /workspace/project
# Project files arrive through antioch.yaml's sync mapping before every
# command runs; they are not build inputs. Install dependencies above this
# line: they are build inputs, and a file this Dockerfile COPYs must have a
# `!path` line in .dockerignore to reach the build.
