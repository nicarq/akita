import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2] / "crates/akita-cpu-backend"


@unittest.skipUnless(
    sys.platform == "darwin" and platform.machine() == "arm64", "requires Apple Silicon"
)
class PackedQuadMetalTests(unittest.TestCase):
    def test_missing_encoder_requests_cpu_fallback(self):
        shaders = BACKEND / "src/opaque/sumcheck"
        source = (
            (shaders / "p64_ext2.metal").read_bytes()
            + (shaders / "relation_range_image/packed_lookup.metal").read_bytes()
            + b"\0"
        )
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "quad_lookup_missing_encoder"
            subprocess.run([
                "xcrun", "clang", "-O2", "-fobjc-arc", "-framework", "Foundation", "-framework", "Metal",
                BACKEND / "tests/quad_lookup_missing_encoder.m", "-o", executable,
            ], check=True)
            run = subprocess.run([executable], input=source, capture_output=True, timeout=60)
        if run.returncode == 77:
            self.skipTest("requires a Metal device")
        self.assertEqual(run.returncode, 0, run.stderr.decode())
