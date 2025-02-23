#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

#
# Push the docker image to docker hub.
#
# Usage: push.sh <USERNAME> <PASSWORD>
#
# PASSWORD: The docker hub account password.
#
DOCKER_HUB_ACCOUNT="$1"

# Get the docker hub account password.
PASSWORD="$2"
shift 1

LOCAL_IMAGE_NAME=allo:latest
REMOTE_IMAGE_NAME_VER=${DOCKER_HUB_ACCOUNT}/allo:latest

echo "Login docker hub"
docker login -u ${DOCKER_HUB_ACCOUNT} -p ${PASSWORD}

echo "Uploading ${LOCAL_IMAGE_NAME} as ${REMOTE_IMAGE_NAME_VER}"
docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE_NAME_VER}
docker push ${REMOTE_IMAGE_NAME_VER}