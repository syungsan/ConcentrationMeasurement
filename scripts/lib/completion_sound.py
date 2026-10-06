"""Play the project's completion sound after a successful job."""
from pathlib import Path
import base64
import os
import shutil
import subprocess
import sys


def _is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _play_wsl_sound(sound: Path) -> None:
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        # IDE terminals can omit Windows directories from PATH.
        candidate = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
        if not candidate.is_file():
            raise RuntimeError("Windows PowerShell が見つかりません。WSL の Windows 相互運用設定を確認してください。")
        powershell = str(candidate)
    # Pass WAV bytes over stdin; Windows need not access a WSL/UNC path.
    # Keep the payload out of the command line (Windows has a length limit).
    script = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "$ErrorActionPreference = 'Stop'; "
        "$stream = New-Object System.IO.MemoryStream; "
        "$player = New-Object System.Media.SoundPlayer; "
        "try { "
        "$bytes = [Convert]::FromBase64String([Console]::In.ReadToEnd()); "
        "$stream.Write($bytes, 0, $bytes.Length); $stream.Position = 0; "
        "$player.Stream = $stream; $player.Load(); $player.PlaySync() "
        "} catch { [Console]::Error.WriteLine($_.Exception.Message); exit 1 } "
        "finally { $player.Dispose(); $stream.Dispose() }"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        input=base64.b64encode(sound.read_bytes()).decode("ascii"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        timeout=60,
    )


def play_completion_sound() -> None:
    sound = Path(__file__).resolve().parents[2] / "mei_kara_mei_switch1.wav"
    try:
        if not sound.is_file():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {sound}")
        if sys.platform == "win32":
            print("[INFO] 完了音を再生します (Windows)", flush=True)
            import winsound
            winsound.PlaySound(str(sound), winsound.SND_FILENAME)
        elif sys.platform == "darwin":
            print("[INFO] 完了音を再生します (macOS)", flush=True)
            subprocess.run(["/usr/bin/afplay", str(sound)], check=True)
        elif _is_wsl():
            print("[INFO] 完了音を再生します (WSL → Windows PowerShell)", flush=True)
            _play_wsl_sound(sound)
        else:
            raise RuntimeError(f"この実行環境では音声再生に対応していません: {sys.platform}")
    except Exception as exc:
        detail = (exc.stderr or "").strip() if isinstance(exc, subprocess.CalledProcessError) else ""
        print(f"[WARN] 音声を再生できませんでした: {detail or exc}", flush=True)


if __name__ == "__main__":
    play_completion_sound()
