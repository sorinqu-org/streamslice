import tempfile
import unittest
from pathlib import Path

from streamslice.metadata import _contains_generic_creator_term, generate_metadata
from streamslice.models import Candidate


class ProfaneMetadataClient:
    def chat_json(self, **_kwargs):
        payload = {
            "title": "Ну блять",
            "description": "Иди нахуй.",
            "hashtags": ["#блядь", "#t2x2", "#clips", "#fyp", "#viral"],
        }
        return payload, "{}"


class MetadataTests(unittest.TestCase):
    def test_rejects_generic_streamer_word_forms(self) -> None:
        for text in (
            "этот стример испугался",
            "со стримером случился фейл",
            "реакция стримера",
            "стримерша засмеялась",
        ):
            with self.subTest(text=text):
                self.assertTrue(_contains_generic_creator_term(text))

    def test_accepts_specific_creator_name(self) -> None:
        self.assertFalse(_contains_generic_creator_term("Тоха не ожидал такого финала"))

    def test_visible_metadata_is_masked_and_profane_hashtag_is_dropped(self) -> None:
        candidate = Candidate(
            0,
            30,
            "test",
            8,
            context_summary="test",
            quality_score=8,
            quality_summary="test",
        )
        with tempfile.TemporaryDirectory() as temp:
            metadata = generate_metadata(
                candidate,
                "блять",
                Path(temp),
                {"models": {"metadata": "stub-model"}},
                ProfaneMetadataClient(),
                creator_identity={"display_name": "t2x2", "preferred_mentions": ["t2x2"]},
            )

        self.assertEqual(metadata["title"], "Ну б***ь")
        self.assertEqual(metadata["description"], "Иди н***й.")
        self.assertNotIn("#блядь", metadata["hashtags"])
