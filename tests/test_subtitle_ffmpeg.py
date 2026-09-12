import sys
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lib.platform_tools import find_subtitle_ffmpeg


class SubtitleFfmpegTests(unittest.TestCase):
    def test_uses_full_build_when_regular_build_lacks_subtitles(self):
        results = [CompletedProcess([], 0, ' .. scale V->V Scale\n'), CompletedProcess([], 0, ' .. subtitles V->V Render text\n')]
        with patch('lib.platform_tools.find_video_tool', return_value=Path('/opt/homebrew/bin/ffmpeg')), patch('lib.platform_tools.sys.platform', 'darwin'), patch.object(Path, 'is_file', return_value=True), patch('lib.platform_tools.subprocess.run', side_effect=results):
            self.assertEqual(find_subtitle_ffmpeg(Path('/repo')), Path('/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg'))

    def test_explains_missing_filter(self):
        with patch('lib.platform_tools.find_video_tool', return_value=Path('/ffmpeg')), patch('lib.platform_tools.sys.platform', 'linux'), patch.object(Path, 'is_file', return_value=True), patch('lib.platform_tools.subprocess.run', return_value=CompletedProcess([], 0, ' .. scale V->V Scale\n')):
            with self.assertRaisesRegex(RuntimeError, 'libass'):
                find_subtitle_ffmpeg(Path('/repo'))

    def test_preserves_supported_windows_build(self):
        with patch('lib.platform_tools.find_video_tool', return_value=Path('/repo/ffmpeg/bin/ffmpeg.exe')), patch('lib.platform_tools.sys.platform', 'win32'), patch.object(Path, 'is_file', return_value=True), patch('lib.platform_tools.subprocess.run', return_value=CompletedProcess([], 0, ' .. subtitles V->V Render text\n')):
            self.assertEqual(find_subtitle_ffmpeg(Path('/repo')), Path('/repo/ffmpeg/bin/ffmpeg.exe'))
