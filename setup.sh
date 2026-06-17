#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Installing via apt..."
  sudo apt update
  sudo apt install -y ffmpeg
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

mkdir -p \
  "$PROJECT_ROOT/footage" \
  "$PROJECT_ROOT/music" \
  "$PROJECT_ROOT/stories" \
  "$PROJECT_ROOT/output" \
  "$PROJECT_ROOT/temp" \
  "$PROJECT_ROOT/modules"

echo
echo "Setup complete."
echo
echo "Next steps:"
echo "1. Drop gameplay/background video files into: $PROJECT_ROOT/footage"
echo "2. Drop royalty-safe background music into: $PROJECT_ROOT/music"
echo "3. Drop Reddit story .txt files into: $PROJECT_ROOT/stories"
echo "4. Set GEMINI_API_KEY, NVIDIA_API_KEY, and optional UNSPLASH_ACCESS_KEY in .env"
echo "5. Launch the web studio: source .venv/bin/activate && python -m webapp.run  (http://127.0.0.1:8000)"
echo "6. Or run from the CLI: python main.py --story stories/your_story.txt"
echo "7. Run everything new: python main.py --all"
echo

if ffmpeg -hide_banner -encoders | grep -q "h264_nvenc"; then
  echo "FFmpeg NVENC encoder is available; GPU rendering will use it when preferred."
else
  echo "FFmpeg NVENC encoder was not found; rendering will fall back to libx264."
fi
