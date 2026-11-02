# Local AI models used by the assistant

All three models run **entirely in the browser** (JavaScript or WebAssembly, no server). The assistant loads each one only
the first time it is needed.

| Folder | Model | Size | Used for | Origin and licence |
|---|---|---|---|---|
| `needle2/` | **Needle 2** (Cactus Compute) | 13.7 MB | AI mode: free-form commands → one app action, chosen from a closed list (finite-grammar decoding) | Pure-JS engine `cact-full.js`, `tokenizer-full.js`, `engine-full.js` from the *Needle 2 Pure-JS* project; model `needle2.cact` from [Cactus-Compute on Hugging Face](https://huggingface.co/Cactus-Compute). `needle-runner.js` and `needle-worker.js` were written for Weka Web. See the upstream model card for the model licence. |
| `whistle/` | **Whistle** (Cactus Compute) | 16.9 MB | Voice commands (speech-to-text: en, de, fr, es, it, nl, pl — not Greek) | Pure-JS runtime `whistle.js` and `whistle_worker.js` from the *Whistle in pure JavaScript* project; `whistle.cact` unchanged from [Cactus-Compute/whistle](https://huggingface.co/Cactus-Compute/whistle). See the upstream model card for the model licence. |
| `orasis/` | **Zagreus Orasis-2** ([zagreus-orasis](https://github.com/AlexandrosGiann/zagreus-orasis)) | 10 MB | Image recognition (1,000 ImageNet labels, about 1,250 words and 14 scene types) and “images → dataset” (1,280-d embeddings) | Runtime and project code under the MIT licence (`LICENSE`). The weights have their own terms — see `NOTICE.txt` and `LICENSE-APACHE-2.0.txt`: they derive from Apache-2.0 models trained on ImageNet-1k, whose terms limit that data to non-commercial research, so get legal advice before commercial use. |

## Requirements

- Orasis-2 is much smaller than MobileCLIP-S0, which it replaces, but less accurate: 54.3% against 69.3% in the project's own
  14-class test with a trained head, and 39.6% against 62.5% from words alone. Treat its labels as rough guesses.
- **AI mode, voice and image recognition work best when the app is served over http(s)**, for example GitHub Pages or
  `python -m http.server` in the app folder, then open <http://localhost:8000>. Browsers block Web Workers and model
  downloads on `file://`.
- Opened from `file://`, the assistant still works (rules, knowledge base, actions). AI mode and voice then ask you to
  choose the model file (`needle2/needle2.cact`, `whistle/whistle.cact`) and run on the main thread; image recognition
  asks for the `orasis` folder.
- The microphone needs `https://` or `localhost`.

## Safety design

- Needle 2 can only output one of the tool calls the app lists (navigate, load a dataset, run a classifier or clusterer,
  find rules, rank attributes, switch theme, export the PDF). It cannot produce free text actions, and **every AI-chosen
  action is shown to the user and runs only after they press “Yes, do it”.**
- Web pages are never fetched automatically by the assistant: it fills in the address on the *Web data* page and the
  user confirms, after robots.txt and rate-limit checks.
