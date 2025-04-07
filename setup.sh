#!/bin/bash

echo "🔧 Setting up AI Vision Assistant..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete. To start the app, run:"
echo "source venv/bin/activate && python main.py"
