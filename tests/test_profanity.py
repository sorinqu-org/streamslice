import unittest

from streamslice.profanity import mask_profanity


class ProfanityMaskTests(unittest.TestCase):
    def test_masks_requested_examples(self) -> None:
        self.assertEqual(mask_profanity("блять"), "б***ь")
        self.assertEqual(mask_profanity("нахуй."), "н***й.")
        self.assertEqual(mask_profanity("нихуя"), "н***я")

    def test_preserves_case_spacing_and_punctuation(self) -> None:
        source = "БЛЯТЬ! Ну, Охуенно...\nЁбаный в рот."
        expected = "Б***Ь! Ну, О*****о...\nЁ****й в рот."
        self.assertEqual(mask_profanity(source), expected)

    def test_masks_common_inflections_and_compounds(self) -> None:
        source = "заебал, пиздец, хуйня, долбоёб, сука и говнище"
        expected = "з****л, п****ц, х***я, д*****б, с**а и г*****е"
        self.assertEqual(mask_profanity(source), expected)

    def test_does_not_mask_neutral_words_containing_similar_substrings(self) -> None:
        source = "страхуй подстрахуй потреблять колебание бляшка хулиган педикюр мандарин"
        self.assertEqual(mask_profanity(source), source)

    def test_masks_only_complete_words_in_mixed_text(self) -> None:
        source = "#блядь, это страховка; 'хуй' — не хулиган."
        expected = "#б***ь, это страховка; 'х*й' — не хулиган."
        self.assertEqual(mask_profanity(source), expected)

    def test_short_and_empty_input(self) -> None:
        self.assertEqual(mask_profanity("ёб бля"), "** б*я")
        self.assertEqual(mask_profanity(""), "")


if __name__ == "__main__":
    unittest.main()
