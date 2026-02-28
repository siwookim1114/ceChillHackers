"""Quick standalone test for MiniMax TTS integration."""

import asyncio
import os
import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
for p in (ROOT, BACKEND):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from services.voice_service import VoiceService


async def main():
    print("=== MiniMax TTS Test ===\n")

    # Check keys
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    group_id = os.environ.get("MINIMAX_GROUP_ID", "")
    if not api_key:
        print("ERROR: MINIMAX_API_KEY not found in .env")
        return
    if not group_id:
        print("ERROR: MINIMAX_GROUP_ID not found in .env")
        return
    print(f"API Key: {api_key[:12]}...{api_key[-4:]}")
    print(f"Group ID: {group_id}\n")

    voice = VoiceService()

    test_text = (
        "Great question! Let's think about eigenvalues step by step. "
        "First, what happens when you multiply a matrix by a vector? "
        "Does the vector always change direction, or can it sometimes just get stretched?"
    )

    print(f"Text to synthesize ({len(test_text)} chars):")
    print(f"  \"{test_text[:80]}...\"\n")
    print("Calling MiniMax TTS API...")

    try:
        audio_bytes = await voice.text_to_speech(test_text)
        print(f"SUCCESS! Got {len(audio_bytes)} bytes of audio\n")

        # Save to file for playback
        output_path = ROOT / "test_output.mp3"
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"Saved to: {output_path}")
        print(f"Play it:  open {output_path}")
        print(f"          # or: afplay {output_path}")

    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
