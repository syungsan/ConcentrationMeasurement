import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from lib.devices import default_device, face_device, face_providers, face_context_id
from lib.platform_tools import find_video_tool


class DeviceTests(unittest.TestCase):
    def test_device_selection(self):
        for cuda, mps, expected in [(True, False, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')]:
            with self.subTest(expected=expected), patch('torch.cuda.is_available', return_value=cuda), patch('torch.backends.mps.is_available', return_value=mps):
                self.assertEqual(default_device(), expected)

    def test_mac_face_backend(self):
        with patch('lib.devices.sys.platform', 'darwin'):
            self.assertEqual(face_device('auto'), 'cpu')
            self.assertEqual(face_device('mps'), 'cpu')
            self.assertEqual(face_device('cuda:0'), 'cuda:0')

    def test_coreml_keeps_acceleration_and_cpu_fallback(self):
        with patch('onnxruntime.get_available_providers', return_value=['CoreMLExecutionProvider', 'CPUExecutionProvider']):
            self.assertEqual(face_device('coreml'), 'coreml')
        self.assertEqual(face_context_id('coreml'), 0)
        self.assertEqual(face_context_id('cpu'), -1)
        providers = face_providers('coreml')
        self.assertEqual(providers[0][0], 'CoreMLExecutionProvider')
        self.assertEqual(providers[0][1]['MLComputeUnits'], 'CPUAndGPU')
        self.assertEqual(providers[-1], 'CPUExecutionProvider')
        with patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider']):
            with self.assertRaisesRegex(RuntimeError, 'CoreML'):
                face_device('coreml')

    def test_native_video_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'ffmpeg' / 'bin'
            folder.mkdir(parents=True)
            (folder / 'ffmpeg.exe').touch()
            with patch('lib.platform_tools.os.name', 'posix'), patch('lib.platform_tools.shutil.which', return_value='/opt/homebrew/bin/ffmpeg'):
                self.assertEqual(str(find_video_tool(root)), '/opt/homebrew/bin/ffmpeg')
            with patch('lib.platform_tools.shutil.which', return_value=None):
                with self.assertRaises(FileNotFoundError):
                    find_video_tool(root, 'ffprobe')
