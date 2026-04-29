from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qwen3_tts_win.core import (
    DEFAULT_CUSTOM_VOICE_MODEL,
    DEFAULT_MODEL,
    DEFAULT_VOICE_DESIGN_MODEL,
    find_reference_in_shared,
    find_reference_text_sidecar,
    resolve_model_for_task,
    resolve_reference_text,
    split_text,
)


class TestCoreHelpers(unittest.TestCase):
    def test_resolve_model_for_task_uses_task_specific_defaults(self):
        self.assertEqual(resolve_model_for_task(DEFAULT_MODEL, "voice_clone"), DEFAULT_MODEL)
        self.assertEqual(resolve_model_for_task(DEFAULT_MODEL, "custom_voice"), DEFAULT_CUSTOM_VOICE_MODEL)
        self.assertEqual(resolve_model_for_task(DEFAULT_MODEL, "voice_design"), DEFAULT_VOICE_DESIGN_MODEL)

    def test_resolve_model_for_task_preserves_explicit_model(self):
        explicit = "Qwen/custom-model"
        self.assertEqual(resolve_model_for_task(explicit, "voice_design"), explicit)

    def test_split_text_respects_max_chars(self):
        text = "First sentence. Second sentence is slightly longer. Third sentence."
        chunks = split_text(text, max_chars=35)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 35 for chunk in chunks))

    def test_find_reference_uses_newest_matching_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared = Path(temp_dir)
            old_ref = shared / "reference_old.wav"
            new_ref = shared / "reference_new.wav"
            old_ref.write_bytes(b"old")
            new_ref.write_bytes(b"new")
            os.utime(old_ref, (1000, 1000))
            os.utime(new_ref, (2000, 2000))

            selected = find_reference_in_shared(shared, "reference")

            self.assertEqual(selected, new_ref.resolve())

    def test_reference_text_sidecar_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared = Path(temp_dir)
            reference = shared / "reference_long.m4a"
            reference.write_bytes(b"audio")
            sidecar = shared / "reference_long.txt"
            sidecar.write_text("  hello   world  ", encoding="utf-8")

            text, source = find_reference_text_sidecar(reference, shared_dir=shared, prefix="reference")

            self.assertEqual(text, "hello world")
            self.assertEqual(source, str(sidecar.resolve()))

    def test_explicit_reference_text_wins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            shared = Path(temp_dir)
            reference = shared / "reference.wav"
            reference.write_bytes(b"audio")
            (shared / "reference.txt").write_text("sidecar", encoding="utf-8")

            text, source = resolve_reference_text(
                explicit_text="  explicit   text  ",
                text_file=None,
                reference_path=reference,
                shared_dir=shared,
                prefix="reference",
            )

            self.assertEqual(text, "explicit text")
            self.assertEqual(source, "explicit text")


if __name__ == "__main__":
    unittest.main()
