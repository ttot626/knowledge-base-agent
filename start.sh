#!/usr/bin/env bash
set -e

echo "Setting up knowledge-base-agent..."
python -m pip install --upgrade pip
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

cat <<'EOF'
Setup complete.
  1) Start database:  docker compose up -d
  2) Copy config:     cp .env.example .env   (then fill DEEPSEEK_API_KEY)
  3) Index documents: python -m kb_agent.embeddings.index_documents
  4) Ask a question:  python -m kb_agent.agent.create_agent
EOF
