"""Interchangeable, content-addressed lexicons; no proprietary lexicon is bundled."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol

from scrabble_gym.environment import load_words


class Lexicon(Protocol):
    @property
    def words(self) -> frozenset[str]: ...

    @property
    def fingerprint(self) -> str: ...


@dataclass(frozen=True)
class WordLexicon:
    words: frozenset[str]

    def __post_init__(self):
        if not self.words or any(
            not word.isascii() or not word.isalpha() or not word.isupper()
            or not 2 <= len(word) <= 15 for word in self.words
        ):
            raise ValueError("Lexicon requires uppercase ASCII words of length 2..15")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256("\n".join(sorted(self.words)).encode("ascii")).hexdigest()

    def __contains__(self, word: str) -> bool:
        return word in self.words


class FileLexicon(WordLexicon):
    def __init__(self, path: str | Path):
        super().__init__(load_words(path))


class TinyLexicon(WordLexicon):
    """Original minimal test vocabulary, not a tournament dictionary."""

    def __init__(self):
        super().__init__(frozenset(
            "AN AS AT BE BY DO GO HE IF IN IS IT ME MY NO OF ON OR OX SO TO UP US WE "
            "ACT ADD AGE AGO AIR ALL AND ANT ANY APE ARE ARM ART ASK ATE BAD BAG BAR "
            "BAT BED BEE BET BIG BIT BOW BOX BOY BUS BUT BUY CAN CAP CAR CAT COW CRY "
            "CUP CUT DAY DID DIE DIG DOG DOT DRY DUE EAR EAT EGG END ERA EYE FAR FAT "
            "FED FEE FEW FIG FIN FIT FIX FLY FOG FOR FOX FUN FUR GAS GET GOD GOT GUM "
            "GUN GUY HAD HAM HAS HAT HAY HER HID HIM HIP HIS HIT HOG HOT HOW HUG HUT "
            "ICE ILL INK ITS IVY JAR JAW JET JOB JOG JOY KEY KID KIT LAB LAD LAP LAW "
            "LAY LED LEG LET LIE LIP LIT LOG LOT LOW MAD MAN MAP MAT MAY MEN MET MIX "
            "MOB MOP MUD MUG NAP NET NEW NOD NOR NOT NOW NUT OAK OAR OAT ODD OFF OIL "
            "OLD ONE OUR OUT OWE OWL OWN PAD PAN PAT PAW PAY PEA PEG PEN PET PIE PIG "
            "PIN PIT POT PUT RAG RAM RAN RAP RAT RAW RAY RED RIB RID RIG RIP ROB ROD "
            "ROT ROW RUB RUG RUN SAD SAG SAP SAT SAW SAY SEA SEE SET SEW SHE SHY SIN "
            "SIP SIR SIT SIX SKI SKY SLY SOB SON SOW SOY SPA SPY SUB SUE SUM SUN TAB "
            "TAG TAN TAP TAR TAX TEA TEN THE TIE TIN TIP TOE TON TOO TOP TOW TOY TRY "
            "TUB TUG TWO USE VAN VAT VET VIE VOW WAR WAS WAX WAY WEB WED WET WHO WHY "
            "WIG WIN WIT WON WOW YES YET YOU ZAP ZIP ZOO CAT CATS DOGS RATE RATES "
            "TEAR TEARS STAR STARS TRAIN TRAINS STONE NOTES TONE TONES".split()
        ))
