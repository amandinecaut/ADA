# ADA Pipeline - Setup & Usage Instructions

## Quick Start

### 1. Initialize the Virtual Environment

```bash
# Create the venv (if not already present)
python3 -m venv .venv_streamlit

# Activate the venv
# macOS / Linux:
source .venv_streamlit/bin/activate

# Windows (PowerShell):
.venv_streamlit\Scripts\Activate.ps1

# Windows (Command Prompt):
.venv_streamlit\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Secrets

Create `.streamlit/secrets.toml` with your API keys (see below). This file is required for LLM integration:

```toml
[settings]
USE_GEMINI = true  # or false for OpenAI

[services.gemini]
GEMINI_API_KEY = "your-gemini-api-key-here"
GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

[services.gpt]
GPT_BASE = "https://api.openai.com/v1"
GPT_KEY = "your-openai-api-key-here"
GPT_ENGINE = "gpt-4o-mini"
```

## Running the App

### Option A: Using the Launch Script (Recommended)

```bash
# Make the script executable (macOS/Linux)
chmod +x run.sh

# Run it
./run.sh
```

The script automatically detects your OS and activates the correct venv path.

### Option B: Manual Activation

**macOS / Linux:**
```bash
source .venv_streamlit/bin/activate
streamlit run src/app.py
```

**Windows (PowerShell):**
```powershell
.venv_streamlit\Scripts\Activate.ps1
streamlit run src/app.py
```

**Windows (Command Prompt):**
```cmd
.venv_streamlit\Scripts\activate
streamlit run src/app.py
```

## Virtual Environment Details

### Primary venv: `.venv_streamlit/`
- **Status**: Active environment for this project
- **Path structure**:
  - macOS/Linux: `.venv_streamlit/bin/activate`
  - Windows: `.venv_streamlit/Scripts/activate`
- **Use case**: All development and production work

### Alternative venvs (Legacy)
- `.venv/` - Older environment (not used)
- `.venv-1/` - Older environment (not used)

**Recommendation**: Use only `.venv_streamlit/` to avoid confusion.

## Verify Installation

After activation and installation, verify everything works:

```bash
# Check Python interpreter
which python

# Check pip packages
pip list | grep -E "streamlit|plotly|scikit-learn|prince"

# Test import
python -c "import streamlit; import plotly; import prince; print('✓ All core modules available')"
```

## Troubleshooting

### Missing Modules
If you see `ModuleNotFoundError` when running the app:
```bash
# Ensure venv is activated (check prompt for (.venv_streamlit))
source .venv_streamlit/bin/activate  # macOS/Linux
# Then reinstall
pip install -r requirements.txt
```

### `.streamlit/secrets.toml` Not Found
The app will fail silently on LLM calls without this file. Create it with your API keys (see section 3 above).

### Streamlit Not Found
```bash
# Verify activation
which streamlit
# Should output: /path/to/.venv/bin/streamlit

# If not found, reinstall
pip install streamlit
```

### Port Already in Use
Streamlit defaults to port 8501. Change it:
```bash
streamlit run src/app.py --server.port 8502
```

## Development Workflow

1. **Always activate** the `.venv_streamlit/` before any work:
   ```bash
   source .venv_streamlit/bin/activate
   ```

2. **Add new dependencies** to `requirements.txt` and reinstall:
   ```bash
   pip install <new-package>
   pip freeze > requirements.txt
   ```

3. **Deactivate** when done:
   ```bash
   deactivate
   ```

## Project Structure

- `src/app.py` - Main Streamlit entry point
- `src/app_utilities.py` - Data loading, FA workflow, clustering
- `src/factor_analysis.py` - FA/FAMD strategy classes
- `src/wordalisation.py` - LLM integrations for labeling/explanation
- `data/demo_data/` - Sample datasets for testing
- `.streamlit/secrets.toml` - **Create this with your API keys**

See `.github/copilot-instructions.md` for detailed architecture documentation.
