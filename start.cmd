@echo off
setlocal

set REPO_NAME=knowledge-base-agent
echo ========================================
echo  Setting up %REPO_NAME%
echo ========================================

echo [1/3] Ensuring pip...
python -m pip install --upgrade pip
if errorlevel 1 echo Failed to upgrade pip

echo [2/3] Creating virtual environment...
python -m venv venv
call .\venv\Scripts\activate
if errorlevel 1 echo Failed to create/activate venv

echo [3/3] Installing package (with dev deps)...
python -m pip install --upgrade pip
pip install -e ".[dev]"
if errorlevel 1 echo Failed to install package

echo ========================================
echo  Setup complete.
echo  1) Start database:  docker compose up -d
echo  2) Copy config:     copy .env.example .env   (then fill DEEPSEEK_API_KEY)
echo  3) Index documents: python -m kb_agent.embeddings.index_documents
echo  4) Ask a question:  python -m kb_agent.agent.create_agent
echo ========================================
endlocal
