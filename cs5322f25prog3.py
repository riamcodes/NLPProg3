import os
from pathlib import Path
from typing import List
import re

import joblib

from wsd_utils import normalize_sentences


def _load_model(word: str):
    """
    Load a pre-trained model for the given word from the 'models' directory.
    """
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / f"{word}_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found for '{word}' at {model_path}. "
            f"Please run the training script to generate models."
        )
    return joblib.load(model_path)


def _predict(word: str, sentences: List[str]) -> List[int]:
    """
    Core predict helper: loads the model for 'word' and returns sense labels (1 or 2).
    """
    if sentences is None:
        raise ValueError("Input must be a list of strings (sentences).")
    model = _load_model(word)
    X = normalize_sentences(sentences)
    preds = model.predict(X)
    # Lightweight heuristic post-processors to boost separation
    if word == "director":
        # Sense 1: Organizational/managerial
        sense1_cues = [
            r"\b(executive|managing|marketing|hospital|regional|operations|finance|technical|program|communications|research|security|customer\s*service)\s+director\b",
            r"\bdirector\s+of\s+(human\s*resources|sales|operations|communications|research|security|customer\s*service|finance|marketing|it|hr)\b",
            r"\b(board|company|organization|department|division|nonprofit|museum|school|hospital)\s+director\b",
            r"\b(chief\s*executive|ceo|board\s+of\s+directors)\b",
        ]
        # Sense 2: Film/theater/media
        sense2_cues = [
            r"\b(film|movie|theater|theatre|cinema|documentary|short\s*film|feature\s*film)\s+director\b",
            r"\b(actor|actress|scene|shot|take|cut|cinematographer|screenwriter|screening|premiere|festival|pre-?production|principal\s+photography)\b",
            r"\bdirector'?s\s+(cut|vision|style|work|previous\s+work|approach)\b",
            r"\b(award-?winning|independent|aspiring|documentary|theater|theatre)\s+director\b",
            r"\b(film\s+school|film\s+festival|standing\s+ovation|visual\s+storytelling|narrative\s+structures)\b",
        ]
        compiled_s1 = [re.compile(pat, flags=re.I) for pat in sense1_cues]
        compiled_s2 = [re.compile(pat, flags=re.I) for pat in sense2_cues]
        adjusted = []
        for s, p in zip(X, preds):
            s2_hit = any(rx.search(s) for rx in compiled_s2)
            s1_hit = any(rx.search(s) for rx in compiled_s1)
            if s2_hit and not s1_hit:
                adjusted.append(2)
            elif s1_hit and not s2_hit:
                adjusted.append(1)
            else:
                adjusted.append(int(p))
        preds = adjusted
    elif word == "rubbish":
        sense1_cues = [
            r"\b(bin|trash|garbage|landfill|dump(ed)?|litter|garbage\s*truck|trash\s*bag|rubbish\s*heap|refuse|debris|waste|scrap)\b",
            r"\b(clean\s*up|dispose|collection|garbage\s*collector|curbside\s*pickup|collection\s*day)\b",
            r"\b(recycle|recycling|waste\s+management|skip\s+bin|rubbish\s*bin|trash\s*can|landfill\s*site|compost|compactor)\b",
            r"\b(tip|dumpster|wheelie\s*bin|refuse\s*site|transfer\s*station|municipal\s*dump)\b",
            r"\b(bag(s)?\s*of\s*rubbish|pile(s)?\s*of\s*rubbish)\b",
        ]
        sense2_cues = [
            r"\b(nonsense|nonsensical|codswallop|tripe|drivel|baloney|bollocks|hogwash|poppycock|claptrap|twaddle|bunkum?|malarkey|guff|piffle|balderdash|flimflam|tommyrot)\b",
            r"(absolute|utter|load of|complete|pile of|lot of|load of old|total|sheer)\s+rubbish",
            r"\btalk(ing)?\s+rubbish\b",
            r"\brubbish!\b",
            r"\b(that'?s|this is)\s+rubbish\b",
            r"\b(rubbish|silly|ridiculous)\s+idea\b",
            r"\bthis\s+article\s+is\s+rubbish\b",
        ]
        compiled_s1 = [re.compile(pat, flags=re.I) for pat in sense1_cues]
        compiled_s2 = [re.compile(pat, flags=re.I) for pat in sense2_cues]
        adjusted = []
        for s, p in zip(X, preds):
            s2_hit = any(rx.search(s) for rx in compiled_s2)
            s1_hit = any(rx.search(s) for rx in compiled_s1)
            if s2_hit and not s1_hit:
                adjusted.append(2)
            elif s1_hit and not s2_hit:
                adjusted.append(1)
            else:
                adjusted.append(int(p))
        preds = adjusted
    # Ensure Python list of ints
    return [int(p) for p in preds]


def WSD_Test_director(sent_list: List[str]) -> List[int]:
    return _predict("director", sent_list)


def WSD_Test_overtime(sent_list: List[str]) -> List[int]:
    return _predict("overtime", sent_list)


def WSD_Test_rubbish(sent_list: List[str]) -> List[int]:
    return _predict("rubbish", sent_list)


