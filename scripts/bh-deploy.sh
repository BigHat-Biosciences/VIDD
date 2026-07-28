#!/bin/bash
# Build and push the vidd-antibody image to ECR.
# Usage:
#   bash scripts/bh-deploy.sh                    # tag = latest, build + push
#   bash scripts/bh-deploy.sh v0.1               # tag = v0.1,   build + push
#   PUSH=0 bash scripts/bh-deploy.sh             # build ONLY, no ECR login, no push
#
# PUSH=0 exists because the vidd-antibody ECR repo does not exist yet (bonobo and
# rerd-antibody do). Creating it is an infra ask -- ECR repos here are declared as
# YAML under bh-ai/stacks/config/<env>/<region>/ecr/. Until it lands, PUSH=0 lets
# the image be built and smoke-tested locally on the util box, which is where all
# the real build risk is anyway.
#
# RUN THIS ON A LINUX x86_64 UTIL BOX, NOT ON A MACBOOK. The image is
# --platform=linux/amd64; cross-building it under emulation on Apple silicon takes
# hours where a util instance takes minutes. bonobo/README.md ("Deploying to ECR")
# is the canonical statement of this. To get to one:
#   bh aws-utility-start      # if it is stopped
#   bh aws-utility            # prints creds + private IP; PEM comes from SSM
#   ssh -i ~/utility.pem ssm-user@<ip>
# then clone/pull VIDD there and run this script.
#
# No --secret id=aws needed: this Dockerfile's only network fetches at build time
# are AF2 weights from storage.googleapis.com and NBB2 weights from zenodo.org,
# neither of which touches S3. If a future layer pulls from S3, copy the IMDS ->
# `--secret id=aws` pattern from wizard_hat/*/tools/openfold3/bh-deploy.sh.
set -euxo pipefail

ECR_URI="332120041740.dkr.ecr.us-east-1.amazonaws.com"
ECR_REPO="$ECR_URI/vidd-antibody"
DOCKER_IMAGE_TAG="${1:-latest}"
PUSH="${PUSH:-1}"

cd "$(dirname "$0")/.."

IMAGE_PATH="${ECR_REPO}:${DOCKER_IMAGE_TAG}"

if [[ "$PUSH" == "1" ]]; then
    aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR_URI"
fi

docker build -f ./Dockerfile --platform=linux/amd64 . -t "$IMAGE_PATH"

if [[ "$PUSH" == "1" ]]; then
    docker push "$IMAGE_PATH"
else
    echo "PUSH=0 -- built ${IMAGE_PATH} locally, not pushing."
fi
