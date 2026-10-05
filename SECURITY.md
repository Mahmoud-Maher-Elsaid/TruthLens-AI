# Security policy

Report security issues privately to the repository owner rather than opening a public issue.

TruthLens AI limits input size, validates extensions and MIME types, sanitizes filenames, constrains CORS, bounds retrieval retries, and returns typed production errors without stack traces or secrets. Environment files, tokens, uploads, vector indexes, checkpoints, and generated datasets are ignored by Git. Uploaded content is processed in memory and is not retained by the API implementation.

Do not place Hugging Face, Qdrant, Modal, ngrok, Vercel, or other credentials in source files. AI verdicts are decision-support outputs and must not be treated as professional legal, medical, financial, or safety advice.
