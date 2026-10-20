# zkie-verified

A bounded, non-ZK fixed-point network proving baseline. This is deliberately
separate from the legacy Transformer/GKR implementation. It does not make that
implementation sound or make its published benchmarks apply to this protocol.

The relation is a 2-input, 2-hidden, 2-output dense network with ReLU between
layers, Q4 scaling and round-half-up. The verifier reads a small public model
manifest, a separately trusted model id, the public statement and proof bytes.
It does not run the network or accept activations/prover data.

Public coefficients are bounded to weights [-8,8], biases [-16,16]; inputs are
[-128,127]. The exact relation and trust boundary are documented in
`docs/security/small-network-protocol.md` at the workspace root.

```sh
cargo +1.94.1 test --release -p zkie-verified --features prover --locked
```

The default build exposes verification only. Enable `prover` to call `evaluate`
or `prove`. Model/statement types support strict JSON; proofs use versioned
binary framing with canonical postcard encoding and a 1 MiB encoded size cap.

No secret setup and no zero-knowledge claim. This small-network baseline has not
received an independent cryptographic audit. It is not a generic ONNX compiler,
Transformer implementation, committed-large-weight verifier or chain verifier.

