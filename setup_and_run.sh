#!/bin/bash
set -e

echo "--- [1/7] Checking Environment ---"
python3 --version

echo "--- [2/7] Creating requirements.txt ---"
cat <<EOF > requirements.txt
torch
transformers
optimum[openvino]
numpy
scipy
librosa
soundfile
google-api-python-client
google-auth-oauthlib
google-auth-httplib2
requests
accelerate
torchaudio
pypdf
python-docx
openpyxl
sentence-transformers
faiss-gpu-cu12
EOF

echo "--- [3/7] Installing Dependencies (System-wide) ---"
pip3 install -r requirements.txt --break-system-packages

echo "--- [4/7] Cleaning old Vector DB ---"
rm -f rag_vector_db.json
echo "Deleted rag_vector_db.json"

echo "--- [5/7] Setting up Directories ---"
mkdir -p rag_docs
if [ -z "$(ls -A rag_docs)" ]; then
   echo "This is a test document about Antigravity system optimization." > rag_docs/test_doc.txt
fi

echo "--- [6/7] Downloading Models ---"
python3 download_model.py

echo "--- [7/7] Verifying RAG System ---"
python3 test_run.py

echo "--- [8/8] Running YouTube SEO Automation (Dry Run) ---"
if [ -f "client_secrets.json" ]; then
    python3 youtube_seo_automation.py --max-videos 1 --log-level DEBUG || echo "YouTube Automation Run Failed (Likely Auth or Config), check logs."
else
    echo "Skipping YouTube Automation: client_secrets.json not found."
    # Create valid dummy secrets if needed? No, let's leave it.
fi

echo "--- DONE ---"

