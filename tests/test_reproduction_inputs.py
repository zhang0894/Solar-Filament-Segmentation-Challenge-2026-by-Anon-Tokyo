import hashlib

import pytest

from scripts.reproduce import validate_test_images


def test_reproduction_rejects_modified_or_missing_test_inputs(tmp_path):
    image = tmp_path / "observation.jpeg"
    image.write_bytes(b"original input fixture")
    manifest = {
        "files": [
            {
                "filename": image.name,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        ]
    }
    validate_test_images(tmp_path, manifest)
    image.write_bytes(b"changed input fixture")
    with pytest.raises(ValueError, match="content differs"):
        validate_test_images(tmp_path, manifest)
    image.unlink()
    with pytest.raises(ValueError, match="filenames differ"):
        validate_test_images(tmp_path, manifest)
