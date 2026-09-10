from __future__ import annotations

import re

# Match letters only. Digits and underscores therefore form hard token boundaries,
# just like punctuation, without becoming part of the masked word.
_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

_EB_PREFIX = r"(?:в|въ|вз|вы|до|за|из|на|над|об|объ|от|пере|под|подъ|по|при|про|раз|рас|с|съ|у)?"
_HU_PREFIX = r"(?:а|в|вы|до|за|на|не|ни|о|об|от|пере|под|по|при|про|раз|рас|с|у)?"
_PIZD_PREFIX = r"(?:без|в|вз|вы|до|за|из|на|о|об|от|пере|под|по|при|про|раз|рас|с|у)?"

# These expressions are matched against the complete normalized word. Keeping
# explicit roots and prefixes avoids substring false positives such as
# "страхуй", "потреблять", "колебание", and "хулиган".
_PROFANE_WORD_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"бля",
        r"бля(?:д|т)[а-я]*",
        (
            rf"{_EB_PREFIX}еб(?:а[а-я]*|е[а-я]*|и[а-я]*|л[а-я]*|н[а-я]*|"
            r"ош[а-я]*|у[а-я]*|ы[а-я]*|ыр[а-я]*|)"
        ),
        r"епт(?:а)?",
        r"(?:долб[ао]|мозго)еб[а-я]*",
        rf"{_HU_PREFIX}(?:ху(?:е[а-я]*|и[а-я]*|й[а-я]*|я[а-я]*)|хул(?:е|и|ь))",
        rf"{_PIZD_PREFIX}пизд[а-я]*",
        r"сук(?:а|е|и|ой|у|ам|ами|ах|ин[а-я]*)",
        r"суч(?:ка|ке|ки|кой|ку|ек|кам|ками|ках)",
        r"(?:мудак|мудач|мудил|мудозвон)[а-я]*",
        r"г[ао]ндон[а-я]*",
        r"п[ие]д[ао]р[а-я]*",
        r"пидр[а-я]*",
        r"педик(?:|а|е|и|ов|ом|у|ам|ами|ах)",
        r"залуп(?:а|е|ой|ою|у|ы|ам|ами|ах)",
        r"(?:вы|за|на|пере|по|под|при|про)?дроч[а-я]*",
        r"жоп(?:а|е|ой|ою|у|ы|ам|ами|ах|н[а-я]*|аст[а-я]*|ош[а-я]*)",
        r"говн[а-я]*",
        r"дерьм[а-я]*",
        r"(?:за)?срач[а-я]*",
        r"срак[а-я]*",
        r"засран[а-я]*",
        r"(?:за|на|об|от|пере|под|по|про|рас|с)?ср(?:ать|ал[а-я]*|ан[а-я]*|ун[а-я]*)",
        r"шалав[а-я]*",
        r"шлюх[а-я]*",
        r"шлюш[а-я]*",
    )
)


def mask_profanity(text: str) -> str:
    """Mask Russian profanity while preserving word boundaries and punctuation.

    The first and last letter stay unchanged for words of at least three
    letters. One- and two-letter profanities are replaced completely because
    they have no interior letter that can carry the required asterisk.
    """

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        normalized = word.casefold().replace("ё", "е")
        if not any(pattern.fullmatch(normalized) for pattern in _PROFANE_WORD_PATTERNS):
            return word
        if len(word) <= 2:
            return "*" * len(word)
        return f"{word[0]}{'*' * (len(word) - 2)}{word[-1]}"

    return _WORD_RE.sub(replace, text)
