#!/usr/bin/env bash
# verify-image-sha256.sh — Verify that a container image matches its expected SHA256 digest.
#
# Usage:
#   ./scripts/verify-image-sha256.sh <image-ref> <expected-sha256>
#
# <image-ref>       Full image reference including tag, e.g.
#                   ghcr.io/nato1000/jessicai-app:1.0.0
# <expected-sha256> The expected digest WITHOUT the "sha256:" prefix, e.g.
#                   abc123...
#
# The script pulls the image manifest digest from the registry (without
# pulling the full image) and compares it against the expected value.
# If the digests match the image is considered safe to deploy.
#
# Requirements: docker (or another OCI-compatible client that supports
#               'docker manifest inspect --verbose'), jq

set -euo pipefail

IMAGE_REF="${1:-}"
EXPECTED_SHA="${2:-}"

if [[ -z "$IMAGE_REF" || -z "$EXPECTED_SHA" ]]; then
  echo "ERROR: Missing arguments." >&2
  echo "Usage: $0 <image-ref> <expected-sha256>" >&2
  exit 1
fi

# Strip an existing sha256: prefix if someone accidentally passes it
EXPECTED_SHA="${EXPECTED_SHA#sha256:}"

echo "[*] Fetching digest for '$IMAGE_REF' from registry..."

if command -v docker &>/dev/null; then
  ACTUAL_SHA=$(docker manifest inspect --verbose "$IMAGE_REF" 2>/dev/null \
    | jq -r '.Descriptor.digest // .digest // empty' \
    | sed 's/^sha256://')
elif command -v crane &>/dev/null; then
  ACTUAL_SHA=$(crane digest "$IMAGE_REF" | sed 's/^sha256://')
elif command -v skopeo &>/dev/null; then
  ACTUAL_SHA=$(skopeo inspect --raw "docker://$IMAGE_REF" \
    | jq -r '.config.digest // empty' \
    | sed 's/^sha256://')
else
  echo "ERROR: No supported container client found (docker, crane, or skopeo)." >&2
  exit 1
fi

if [[ -z "$ACTUAL_SHA" ]]; then
  echo "ERROR: Could not retrieve digest for '$IMAGE_REF'." >&2
  exit 1
fi

echo "[*] Expected: sha256:$EXPECTED_SHA"
echo "[*] Actual:   sha256:$ACTUAL_SHA"

if [[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]]; then
  echo "[+] SHA256 digest MATCHES — image is authentic."
else
  echo "[-] SHA256 digest MISMATCH — image may have been tampered with!" >&2
  exit 2
fi
