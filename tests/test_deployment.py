import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from omnifactory.data.schema import CanonicalSample
from omnifactory.deployment import check_server, serving_messages
from omnifactory.errors import OmniFactoryError


def test_http_check_roundtrip_and_model_identity(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "test-model"}]}).encode())

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        result = check_server(url, "test-model", output=tmp_path / "report.json")
        assert result["requests"] == 1 and result["speech_generation_verified"] is False
        assert received[0]["modalities"] == ["text"]
        with pytest.raises(OmniFactoryError, match="E_SERVE_MODEL"):
            check_server(url, "missing")
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_serving_uses_only_history(raw_sample, tmp_path):
    # Replace audio context with text so no files are needed for this transport test.
    raw_sample["messages"][0]["content"] = [{"id": "prompt", "type": "text", "text": "question"}]
    sample = CanonicalSample.model_validate(raw_sample)
    messages = serving_messages(sample, "p0", tmp_path)
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "question"}]}]
