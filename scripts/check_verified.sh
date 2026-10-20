#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# All work is bounded to a two-layer 2x2 network; no model download or benchmark.
cargo +1.94.1 fmt -p zkie-verified --check
cargo +1.94.1 test --release -p zkie-verified --features prover --locked
cargo +1.94.1 clippy --release -p zkie-verified --all-targets --features prover --locked -- -D warnings
cargo +1.94.1 build --release -p zkie-verified --features prover --bins --locked
# A separate target directory prevents accidentally testing the prover-enabled
# library artifact under the name of a verifier-only build.
cargo +1.94.1 build --release -p zkie-verified --no-default-features \
    --bin zkie-verify --target-dir target/verified-verifier --locked
python3 scripts/accept_verified.py target/release/zkie-prove \
    target/verified-verifier/release/zkie-verify
