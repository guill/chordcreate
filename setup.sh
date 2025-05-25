#!/bin/bash
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
    echo "Activating virtual environment..."
fi
pip install -r requirements.txt
python -m spacy download en_core_web_md
