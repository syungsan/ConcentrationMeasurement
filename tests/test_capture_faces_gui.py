import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

path = Path(__file__).resolve().parents[1] / 'scripts/tools/capture_faces_gui.py'
spec = importlib.util.spec_from_file_location('capture_faces_gui', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CaptureTests(unittest.TestCase):
    def test_camera_released_when_window_creation_fails(self):
        cap = MagicMock()
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'FaceAnalysis'), patch.object(module.cv2, 'VideoCapture', return_value=cap), patch.object(module.cv2, 'namedWindow', side_effect=RuntimeError('window failed')), patch.object(module.cv2, 'destroyAllWindows') as destroy:
            with self.assertRaisesRegex(RuntimeError, 'window failed'):
                module.capture_faces(0, directory, 'test', device='cpu')
            cap.release.assert_called_once()
            destroy.assert_called_once()

    def test_stop_event_prevents_read(self):
        cap = MagicMock()
        stop = MagicMock()
        stop.is_set.return_value = True
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'FaceAnalysis') as factory, patch.object(module.cv2, 'VideoCapture', return_value=cap), patch.object(module.cv2, 'namedWindow'), patch.object(module.cv2, 'destroyAllWindows'), patch.object(module.os, 'name', 'posix'):
            module.capture_faces(0, directory, 'test', device='cpu', stop_event=stop)
            self.assertEqual(factory.call_args.kwargs['allowed_modules'], ['detection'])
            cap.read.assert_not_called()
            cap.release.assert_called_once()

    def test_worker_reports_error(self):
        results = MagicMock()
        with patch.object(module, 'capture_faces', side_effect=RuntimeError('camera failed')), patch.object(module.traceback, 'print_exc'):
            module.capture_worker({}, None, results)
        results.put.assert_called_once_with('camera failed')
