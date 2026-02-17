
#!/bin/bash
# ADA Pipeline - Streamlit App Launcher
# Uses .venv_streamlit (the primary venv for this project)

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv_streamlit/Scripts/activate
else
    # macOS and Linux
    source .venv_streamlit/bin/activate
fi

streamlit run ./src/app.py
