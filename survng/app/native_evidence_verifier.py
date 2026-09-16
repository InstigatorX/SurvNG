"""Own one CPU gvadetect verifier, isolated from the live GPU process."""
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time

from .dlstreamer_capture import adjacent_model_proc, live_python_executable


class NativeEvidenceVerifier:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.process = None
        self.directory = None
        self.error_file = None
        self.settings = None

    def _close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdin.close()
            self.process = None
        if self.error_file is not None:
            self.error_file.close()
            self.error_file = None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    def request_stop(self):
        self.stopping.set()

    def close(self):
        self.request_stop()
        with self.lock:
            self._close()

    def detect(self, image):
        with self.lock:
            if self.stopping.is_set():
                raise RuntimeError("native evidence verifier stopped")
            model = str(Path(self.config.resolved_model_path()).resolve())
            settings = {
                "model": model,
                "model_proc": adjacent_model_proc(model),
                "nms": self.config.nms_threshold,
                "threshold": min(self.config.confidence_threshold, 0.35,
                                 *self.config.event_class_confidence_thresholds.values()),
                "labels_path": self.config.labels_path,
                "labels": self.config.labels,
            }
            if self.process is None or self.process.poll() is not None or settings != self.settings:
                self._close()
                self.directory = tempfile.TemporaryDirectory(prefix="survng-native-evidence-")
                root = Path(self.directory.name)
                (root/"config.json").write_text(json.dumps(settings))
                self.settings = settings
                self.error_file = (root/"stderr.log").open("w")
                self.process = subprocess.Popen([live_python_executable(),"-m","survng.native_evidence_verify",str(root/"config.json")],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.error_file,text=True,cwd=Path(__file__).resolve().parents[2])
            root = Path(self.directory.name)
            result = root/"result.json"
            result.unlink(missing_ok=True)
            pixels = root/"frame.bgr"
            pixels.write_bytes(image.tobytes())
            self.process.stdin.write(json.dumps({"width":image.shape[1],"height":image.shape[0],"pixels":str(pixels),"result":str(result)})+"\n")
            self.process.stdin.flush()
            deadline = time.monotonic()+40
            while not result.exists():
                if self.stopping.is_set() or time.monotonic()>deadline or self.process.poll() is not None:
                    detail = (root/"stderr.log").read_text()[-1600:]
                    self._close()
                    raise RuntimeError("native evidence verifier unavailable or timed out: " + detail)
                time.sleep(0.02)
            payload=json.loads(result.read_text())
            if payload.get("error"):
                self._close()
                raise RuntimeError(payload["error"])
            return payload.get("objects",[])
