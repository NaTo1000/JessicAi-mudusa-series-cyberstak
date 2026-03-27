#!/usr/bin/env bash
# verify-manifest.sh — Verify the GPG signature of a Kubernetes manifest.
#
# Usage:
#   ./scripts/verify-manifest.sh <manifest.yaml> [signature.asc]
#
# By default the script looks for <manifest.yaml>.asc as the signature file.
#
# Requirements: gpg

set -euo pipefail

MANIFEST="${1:-}"
SIG_FILE="${2:-${MANIFEST}.asc}"

if [[ -z "$MANIFEST" ]]; then
  echo "ERROR: No manifest file specified." >&2
  echo "Usage: $0 <manifest.yaml> [signature.asc]" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: Manifest not found: $MANIFEST" >&2
  exit 1
fi

if [[ ! -f "$SIG_FILE" ]]; then
  echo "ERROR: Signature file not found: $SIG_FILE" >&2
  exit 1
fi

echo "[*] Verifying GPG signature for '$MANIFEST'..."
if gpg --batch --verify "$SIG_FILE" "$MANIFEST"; then
  echo "[+] Signature VALID — manifest integrity confirmed."
else
  echo "[-] Signature INVALID — manifest may have been tampered with!" >&2
  exit 2
fi
