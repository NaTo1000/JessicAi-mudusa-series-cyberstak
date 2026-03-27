#!/usr/bin/env bash
# sign-manifest.sh — GPG-sign a Kubernetes manifest file.
#
# Usage:
#   ./scripts/sign-manifest.sh <manifest.yaml> [GPG_KEY_ID]
#
# The script produces a detached ASCII-armored signature file alongside
# the manifest (e.g., deployment.yaml.asc) and also patches the
# security.jessicai/gpg-signature annotation in the manifest itself
# using 'kubectl annotate' when the target cluster is reachable.
#
# Requirements: gpg, kubectl (optional for live annotation)

set -euo pipefail

MANIFEST="${1:-}"
GPG_KEY="${2:-${JESSICAI_GPG_KEY_ID:-}}"

if [[ -z "$MANIFEST" ]]; then
  echo "ERROR: No manifest file specified." >&2
  echo "Usage: $0 <manifest.yaml> [GPG_KEY_ID]" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: File not found: $MANIFEST" >&2
  exit 1
fi

if [[ -z "$GPG_KEY" ]]; then
  echo "ERROR: GPG key ID not specified." >&2
  echo "Set the JESSICAI_GPG_KEY_ID environment variable or pass it as the second argument." >&2
  exit 1
fi

SIG_FILE="${MANIFEST}.asc"

echo "[*] Signing '$MANIFEST' with GPG key '$GPG_KEY'..."
gpg --batch --yes \
    --local-user "$GPG_KEY" \
    --armor \
    --detach-sign \
    --output "$SIG_FILE" \
    "$MANIFEST"

echo "[+] Signature written to '$SIG_FILE'"

# Optionally embed the signature as a Kubernetes annotation
if command -v kubectl &>/dev/null && kubectl cluster-info &>/dev/null 2>&1; then
  # Extract resource kind/name from the manifest for kubectl annotate
  KIND=$(grep -m1 '^kind:' "$MANIFEST" | awk '{print $2}' | tr '[:upper:]' '[:lower:]')
  NAME=$(grep -m1 '^  name:' "$MANIFEST" | awk '{print $2}')
  NAMESPACE=$(grep -m1 'namespace:' "$MANIFEST" | awk '{print $2}' || echo "jessicai")

  if [[ -n "$KIND" && -n "$NAME" ]]; then
    SIG_B64=$(base64 < "$SIG_FILE" | tr -d '\n')
    echo "[*] Annotating $KIND/$NAME in namespace $NAMESPACE..."
    kubectl annotate "$KIND" "$NAME" \
      -n "$NAMESPACE" \
      "security.jessicai/gpg-signature=$SIG_B64" \
      --overwrite
    echo "[+] Annotation applied."
  fi
fi

echo "[+] Done."
