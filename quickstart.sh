#!/bin/bash
# Quick start script for lyrics emotion transfer project

echo "Cross-Genre Emotion Classification - Quick Start"
echo "================================================"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found"
    echo ""
    echo "To get started:"
    echo "1. Copy .env.example to .env:"
    echo "   cp .env.example .env"
    echo ""
    echo "2. Visit https://genius.com/api-clients"
    echo "3. Create a new API client"
    echo "4. Copy your Client Access Token"
    echo "5. Edit .env and add your token:"
    echo "   GENIUS_API_TOKEN=your_token_here"
    echo ""
    exit 1
fi

echo "✓ .env file found"

# Check if token is set in .env
if ! grep -q "GENIUS_API_TOKEN=your_genius_api_token_here" .env && grep -q "GENIUS_API_TOKEN=" .env; then
    echo "✓ GENIUS_API_TOKEN configured in .env"
else
    echo "⚠  Please edit .env and add your Genius API token"
    exit 1
fi
echo ""

# Check if NRC lexicon exists
if [ ! -f "data/raw/NRC-VAD-Lexicon.txt" ]; then
    echo "⚠  NRC-VAD Lexicon not found"
    echo ""
    echo "To download:"
    echo "1. Visit https://saifmohammad.com/WebPages/nrc-vad.html"
    echo "2. Download NRC-VAD-Lexicon.txt"
    echo "3. Save to data/raw/NRC-VAD-Lexicon.txt"
    echo ""
    echo "Note: Emotion annotation will use example lexicon until this is added."
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Create necessary directories
mkdir -p data/raw/genius_pulls/{pop,hiphop,country,rock}
mkdir -p data/processed
mkdir -p data/splits
mkdir -p results/{models,metrics,features,figures}

echo "✓ Directory structure created"
echo ""

echo "Ready to collect data!"
echo ""
echo "Next steps:"
echo "1. cd src/data"
echo "2. python collection.py  (collects ~8000 songs, takes 4-5 hours)"
echo "3. python emotion_annotation.py  (annotates with emotions)"
echo ""
echo "Or run interactively in Python:"
echo "  from collection import GeniusCollector"
echo "  collector = GeniusCollector(os.getenv('GENIUS_API_TOKEN'))"
echo "  collector.collect_genre('pop', max_songs_per_artist=10)  # small test"
echo ""