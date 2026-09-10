"""Play the project's completion sound after a successful job."""
from pathlib import Path
import subprocess
import sys


def play_completion_sound() -> None:
    sound = Path(__file__).resolve().parents[2] / "mei_kara_mei_switch1.wav"
    try:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(str(sound), winsound.SND_FILENAME)
        elif sys.platform == "darwin":
            subprocess.run(["/usr/bin/afplay", str(sound)], check=True)
    except Exception as exc:
        print(f"[WARN] 音声を再生できませんでした: {exc}")
