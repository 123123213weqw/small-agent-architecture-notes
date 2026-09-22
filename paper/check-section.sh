#!/usr/bin/env bash
# Compile a single paper section on its own, to catch LaTeX errors early.
#
#   ./check-section.sh architecture-comparison-plan
#
# Writes a throwaway wrapper, compiles it with tectonic, removes the wrapper.
set -euo pipefail

name="${1:?usage: check-section.sh <section-basename>}"
cd "$(dirname "$0")"

if [ ! -f "sections/${name}.tex" ]; then
  echo "no such section: sections/${name}.tex" >&2
  exit 2
fi

tmp="_check-${name}.tex"
mkdir -p /tmp/tectonic-check
cat > "$tmp" <<EOF
\\documentclass[11pt,a4paper]{article}
\\input{preamble}
\\begin{document}
\\input{sections/${name}}
\\end{document}
EOF

trap 'rm -f "$tmp"' EXIT
tectonic -X compile "$tmp" --outdir /tmp/tectonic-check --keep-logs
