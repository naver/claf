#!/usr/bin/env python
# encoding: utf-8

"""
Hangulpy.py
Copyright (C) 2012 Ryan Rho, Hyunwoo Cho
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import string
import re

################################################################################
# Hangul Unicode Variables
################################################################################

# Code = 0xAC00 + (Chosung_index * NUM_JOONGSUNGS * NUM_JONGSUNGS) + (Joongsung_index * NUM_JONGSUNGS) + (Jongsung_index)
CHOSUNGS = [
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]
JOONGSUNGS = [
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]
JONGSUNGS = [
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]

NUM_CHOSUNGS = 19
NUM_JOONGSUNGS = 21
NUM_JONGSUNGS = 28

FIRST_HANGUL_UNICODE = 0xAC00  # '가'
LAST_HANGUL_UNICODE = 0xD7A3  # '힣'

################################################################################
# Boolean Hangul functions
################################################################################


def is_hangul(phrase):  # pragma: no cover
    """Check whether the phrase is Hangul.
    This method ignores white spaces, punctuations, and numbers.
    @param phrase a target string
    @return True if the phrase is Hangul. False otherwise."""

    # If the input is only one character, test whether the character is Hangul.
    if len(phrase) == 1:
        return is_all_hangul(phrase)

    # Remove all white spaces, punctuations, numbers.
    exclude = set(string.whitespace + string.punctuation + "0123456789")
    phrase = "".join(ch for ch in phrase if ch not in exclude)

    return is_all_hangul(phrase)


def is_all_hangul(phrase):  # pragma: no cover
    """Check whether the phrase contains all Hangul letters
    @param phrase a target string
    @return True if the phrase only consists of Hangul. False otherwise."""

    for unicode_value in map(lambda letter: ord(letter), phrase):
        if unicode_value < FIRST_HANGUL_UNICODE or unicode_value > LAST_HANGUL_UNICODE:
            # Check whether the letter is chosungs, joongsungs, or jongsungs.
            if unicode_value not in map(lambda v: ord(v), CHOSUNGS + JOONGSUNGS + JONGSUNGS[1:]):
                return False
    return True


def has_jongsung(letter):  # pragma: no cover
    """Check whether this letter contains Jongsung"""
    if len(letter) != 1:
        raise Exception("The target string must be one letter.")
    if not is_hangul(letter):
        raise NotHangulException("The target string must be Hangul")

    unicode_value = ord(letter)
    return (unicode_value - FIRST_HANGUL_UNICODE) % NUM_JONGSUNGS > 0


def has_batchim(letter):  # pragma: no cover
    """This method is the same as has_jongsung()"""
    return has_jongsung(letter)


def has_approximant(letter):  # pragma: no cover
    """Approximant makes complex vowels, such as ones starting with y or w.
    In Korean there is a unique approximant euㅡ making uiㅢ, but ㅢ does not make many irregularities."""
    if len(letter) != 1:
        raise Exception("The target string must be one letter.")
    if not is_hangul(letter):
        raise NotHangulException("The target string must be Hangul")

    jaso = decompose(letter)
    diphthong = (2, 3, 6, 7, 9, 10, 12, 14, 15, 17)
    # [u'ㅑ',u'ㅒ',',u'ㅕ',u'ㅖ',u'ㅘ',u'ㅙ',u'ㅛ',u'ㅝ',u'ㅞ',u'ㅠ']
    # excluded 'ㅢ' because y- and w-based complex vowels are irregular.
    # vowels with umlauts (ㅐ, ㅔ, ㅚ, ㅟ) are not considered complex vowels.
    return jaso[1] in diphthong


################################################################################
# Decomposition & Combination
################################################################################


def compose(chosung, joongsung, jongsung=""):  # pragma: no cover
    """This function returns a Hangul letter by composing the specified chosung, joongsung, and jongsung.
    @param chosung
    @param joongsung
    @param jongsung the terminal Hangul letter. This is optional if you do not need a jongsung."""

    if jongsung is None:
        jongsung = ""

    try:
        chosung_index = CHOSUNGS.index(chosung)
        joongsung_index = JOONGSUNGS.index(joongsung)
        jongsung_index = JONGSUNGS.index(jongsung)
    except Exception as e:
        raise NotHangulException(
            "No valid Hangul character can be generated using given combination of chosung, joongsung, and jongsung."
        )

    return chr(
        0xAC00
        + chosung_index * NUM_JOONGSUNGS * NUM_JONGSUNGS
        + joongsung_index * NUM_JONGSUNGS
        + jongsung_index
    )


def decompose(hangul_letter):  # pragma: no cover
    """This function returns letters by decomposing the specified Hangul letter."""

    if len(hangul_letter) < 1:
        raise NotLetterException("")
    elif not is_hangul(hangul_letter):
        raise NotHangulException("")

    code = ord(hangul_letter) - FIRST_HANGUL_UNICODE
    jongsung_index = int(code % NUM_JONGSUNGS)
    code /= NUM_JONGSUNGS
    joongsung_index = int(code % NUM_JOONGSUNGS)
    code /= NUM_JOONGSUNGS
    chosung_index = int(code)

    return (CHOSUNGS[chosung_index], JOONGSUNGS[joongsung_index], JONGSUNGS[jongsung_index])


################################################################################
# Josa functions
################################################################################


def josa_en(word):  # pragma: no cover
    """add josa either '은' or '는' at the end of this word"""
    word = word.strip()
    if not is_hangul(word):
        raise NotHangulException("")

    last_letter = word[-1]
    josa = "은" if has_jongsung(last_letter) else "는"
    return word + josa


def josa_eg(word):  # pragma: no cover
    """add josa either '이' or '가' at the end of this word"""
    word = word.strip()
    if not is_hangul(word):
        raise NotHangulException("")

    last_letter = word[-1]
    josa = "이" if has_jongsung(last_letter) else "가"
    return word + josa


def josa_el(word):  # pragma: no cover
    """add josa either '을' or '를' at the end of this word"""
    word = word.strip()
    if not is_hangul(word):
        raise NotHangulException("")

    last_letter = word[-1]
    josa = "을" if has_jongsung(last_letter) else "를"
    return word + josa


def josa_ro(word):  # pragma: no cover
    """add josa either '으로' or '로' at the end of this word"""
    word = word.strip()
    if not is_hangul(word):
        raise NotHangulException("")

    last_letter = word[-1]
    if not has_jongsung(last_letter):
        josa = "로"
    elif (ord(last_letter) - FIRST_HANGUL_UNICODE) % NUM_JONGSUNGS == 9:  # ㄹ
        josa = "로"
    else:
        josa = "으로"

    return word + josa


def josa_gwa(word):  # pragma: no cover
    """add josa either '과' or '와' at the end of this word"""
    word = word.strip()
    if not is_hangul(word):
        raise NotHangulException("")

    last_letter = word[-1]
    josa = "과" if has_jongsung(last_letter) else "와"
    return word + josa


def josa_ida(word):  # pragma: no cover
    """add josa either '이다' or '다' at the end of this word"""
    word = word.strip()
    if not is_hangul(word):
        raise NotHangulException("")

    last_letter = word[-1]
    josa = "이다" if has_jongsung(last_letter) else "다"
    return word + josa


################################################################################
# Prefixes and suffixes
# Practice area; need more organization
################################################################################


def add_ryul(word):  # pragma: no cover
    """add suffix either '률' or '율' at the end of this word"""
    word = word.strip()
    if not is_hangul(word):
        raise NotHangulException("")

    last_letter = word[-1]
    if not has_jongsung(last_letter):
        ryul = "율"
    elif (ord(last_letter) - FIRST_HANGUL_UNICODE) % NUM_JONGSUNGS == 4:  # ㄴ
        ryul = "율"
    else:
        ryul = "률"

    return word + ryul


################################################################################
# The formatter, or ultimately, a template system
# Practice area; need more organization
################################################################################


def ili(word):  # pragma: no cover
    """convert {가} or {이} to their correct respective particles automagically."""
    word = word.strip()
    if not is_hangul(word):
        raise NotHangulException("")

    last_letter = word[word.find("{가}") - 1]
    word = word.replace("{가}", ("이" if has_jongsung(last_letter) else "가"))

    last_letter = word[word.find("{이}") - 1]
    word = word.replace("{이}", ("이" if has_jongsung(last_letter) else "가"))
    return word


################################################################################
# Exceptions
################################################################################


class NotHangulException(Exception):  # pragma: no cover
    pass


class NotLetterException(Exception):  # pragma: no cover
    pass


class NotWordException(Exception):  # pragma: no cover
    pass
