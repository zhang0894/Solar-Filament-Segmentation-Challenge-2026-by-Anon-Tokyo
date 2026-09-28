import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts.download_checkpoints import download_archive


@pytest.mark.parametrize("ranges,invalid", [(True, False), (False, False), (True, True)])
def test_checkpoint_transfer_preserves_bytes_or_rejects_bad_coverage(
    tmp_path, monkeypatch, ranges, invalid
):
    payload = bytes(range(256)) * 5000
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setattr("scripts.download_checkpoints.time.sleep", lambda _: None)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requested = self.headers.get("Range")
            start, end = 0, len(payload) - 1
            if requested:
                start, end = map(int, requested.removeprefix("bytes=").split("-"))
            self.send_response(206 if requested else 200)
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            if requested:
                total = len(payload) + int(invalid)
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            try:
                self.wfile.write(payload[start : end + 1])
            except ConnectionError:
                pass  # The initial metadata probe deliberately closes its body.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = tmp_path / "bundle.partial"
    try:
        url = f"http://127.0.0.1:{server.server_port}/weights"
        if invalid:
            with pytest.raises(ValueError, match="unexpected byte range"):
                download_archive(url, target, {}, workers=4)
        else:
            download_archive(url, target, {}, workers=4)
            assert target.read_bytes() == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
