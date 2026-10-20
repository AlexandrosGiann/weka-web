# Model attribution and derivative disclosure

> Apple Machine Learning Research Model is licensed under the Apple Machine
> Learning Research Model License Agreement.

This repository uses MobileCLIP-S0 model material originating from Apple's
ML-MobileCLIP project and the ONNX conversion published as
`Xenova/mobileclip_s0`.

The browser representation in this repository is a derivative/converted
artifact for research and academic use:

- `compiled-fp32-v1/weights.bin` contains serialized FP32 static tensors from
  the MobileCLIP-S0 vision ONNX graph.
- `compiled-fp32-v1/plan.json` is a browser-oriented execution plan generated
  from that graph, with Constant nodes folded.
- `text-bank/embeddings.f32` contains precomputed MobileCLIP text embeddings
  for the labels in `text-bank/labels.json`.
- No retraining or fine-tuning was performed.
- The runtime implements inference in pure JavaScript TypedArrays.

The model files are subject to Apple's model license, not the repository's MIT
software license. See:

- `licenses/LICENSE_MODELS_APPLE`
- `licenses/LICENSE_XENOVA_MOBILECLIP_S0`

Use and redistribution of the model material must comply with those terms.
