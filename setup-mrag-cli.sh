#!/usr/bin/env bash
# One-time setup: symlink mragb and mragf into ~/bin so you can run them from any directory.
# Run: ./setup-mrag-cli.sh  (or: bash setup-mrag-cli.sh)

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="$HOME/bin"
mkdir -p "$BIN"
ln -sf "$ROOT/mragb" "$BIN/mragb"
ln -sf "$ROOT/mragf" "$BIN/mragf"
ln -sf "$ROOT/mragm" "$BIN/mragm"
ln -sf "$ROOT/mragt" "$BIN/mragt"
ln -sf "$ROOT/mragw" "$BIN/mragw"
echo "Linked mragb, mragf, mragm, mragt, mragw to $BIN"
if ! echo ":$PATH:" | grep -q ":$BIN:"; then
  echo ""
  echo "Add $BIN to your PATH. For zsh, add this to ~/.zshrc:"
  echo "  export PATH=\"\$HOME/bin:\$PATH\""
  echo "Then run: source ~/.zshrc"
fi
