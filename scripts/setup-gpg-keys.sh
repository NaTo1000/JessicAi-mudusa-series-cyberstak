#!/usr/bin/env bash
# setup-gpg-keys.sh — Bootstrap GPG keys for the JessicAi Kubernetes environment.
#
# This script generates (or imports) the signing key pair used to sign
# Kubernetes manifests and optionally exports the public key for
# distribution to cluster operators.
#
# Usage (generate a new key):
#   ./scripts/setup-gpg-keys.sh generate [KEY_NAME] [KEY_EMAIL]
#
# Usage (import an existing secret key):
#   ./scripts/setup-gpg-keys.sh import <secret-key.asc>
#
# Usage (export the public key for distribution):
#   ./scripts/setup-gpg-keys.sh export [KEY_ID]
#
# Requirements: gpg >= 2.1

set -euo pipefail

COMMAND="${1:-generate}"
KEY_NAME="${2:-JessicAi Kubernetes Signing Key}"
KEY_EMAIL="${3:-kubernetes-signing@jessicai.local}"

case "$COMMAND" in

  generate)
    echo "[*] Generating GPG key for '$KEY_NAME <$KEY_EMAIL>'..."
    gpg --batch --gen-key <<EOF
%no-protection
Key-Type: EdDSA
Key-Curve: ed25519
Subkey-Type: ECDH
Subkey-Curve: cv25519
Name-Real: $KEY_NAME
Name-Email: $KEY_EMAIL
Expire-Date: 1y
%commit
EOF
    KEY_ID=$(gpg --list-keys --with-colons "$KEY_EMAIL" \
      | awk -F: '/^pub/{print $5}' | tail -1)
    echo "[+] Key generated. Key ID: $KEY_ID"
    echo "[*] Exporting public key to 'jessicai-signing-pub.asc'..."
    gpg --armor --export "$KEY_EMAIL" > jessicai-signing-pub.asc
    echo "[+] Public key saved to 'jessicai-signing-pub.asc'. Distribute this to all verifiers."
    ;;

  import)
    KEY_FILE="${2:-}"
    if [[ -z "$KEY_FILE" || ! -f "$KEY_FILE" ]]; then
      echo "ERROR: Please provide a valid secret key file to import." >&2
      echo "Usage: $0 import <secret-key.asc>" >&2
      exit 1
    fi
    echo "[*] Importing GPG secret key from '$KEY_FILE'..."
    gpg --batch --import "$KEY_FILE"
    echo "[+] Key imported."
    ;;

  export)
    KEY_ID="${2:-$KEY_EMAIL}"
    echo "[*] Exporting public key for '$KEY_ID'..."
    gpg --armor --export "$KEY_ID" > jessicai-signing-pub.asc
    echo "[+] Public key exported to 'jessicai-signing-pub.asc'."
    ;;

  *)
    echo "ERROR: Unknown command '$COMMAND'." >&2
    echo "Usage: $0 {generate|import|export} [args...]" >&2
    exit 1
    ;;
esac
