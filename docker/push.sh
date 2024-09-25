#!/usr/bin/env bash
# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

#
# Push the docker image to docker hub.
#
# Usage: push.sh <PASSWORD>
#
# PASSWORD: The docker hub account password.
#
DOCKER_HUB_ACCOUNT=chhzh123

# Get the docker hub account password.
PASSWORD="$1"
shift 1

LOCAL_IMAGE_NAME=hcl-dialect:latest
REMOTE_IMAGE_NAME_VER=${DOCKER_HUB_ACCOUNT}/hcl-dialect:llvm-18.x-py3.12

echo "Login docker hub"
docker login -u ${DOCKER_HUB_ACCOUNT} -p ${PASSWORD}

echo "Uploading ${LOCAL_IMAGE_NAME} as ${REMOTE_IMAGE_NAME_VER}"
docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_NAME_VER}
docker push ${REMOTE_IMAGE_NAME_VER}