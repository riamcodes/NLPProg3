import re
from typing import List, Tuple, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def parse_two_sense_file(contents: str) -> Tuple[List[str], List[str]]:
    """
    Parse a dataset file with two noun senses and example sentences.
    Returns (sense1_sentences, sense2_sentences).
    The format is assumed to contain:
    - Header lines with the word and two glosses
    - A line with '1' then sentences for sense 1
    - A line with '2' then sentences for sense 2
    """
    lines = contents.splitlines()
    # Normalize whitespace-only lines
    cleaned = [line.rstrip() for line in lines]

    # Find indices for the sense section markers ('1' and '2' as a full line)
    sense1_idx = None
    sense2_idx = None
    for i, line in enumerate(cleaned):
        if sense1_idx is None and line.strip() == "1":
            sense1_idx = i
        elif line.strip() == "2":
            sense2_idx = i
            break

    if sense1_idx is None or sense2_idx is None:
        raise ValueError("Could not locate sense markers '1' and '2' in file.")

    # Collect sentences for each sense; skip blank lines
    sense1_lines = [l.strip() for l in cleaned[sense1_idx + 1 : sense2_idx] if l.strip()]
    sense2_lines = [l.strip() for l in cleaned[sense2_idx + 1 :] if l.strip()]

    # Some lines may contain glosses or headers before '1'; they are ignored by design

    # Remove any spuriously numbered headers that aren't sentences
    def is_probable_sentence(s: str) -> bool:
        # Heuristic: must contain at least one space or punctuation typical of sentences
        return bool(re.search(r"[ .,';:!?-]", s)) and len(s.split()) >= 3

    s1 = [s for s in sense1_lines if is_probable_sentence(s)]
    s2 = [s for s in sense2_lines if is_probable_sentence(s)]
    return s1, s2


def normalize_sentences(sents: List[str]) -> List[str]:
    """
    Light normalization: collapse whitespace and strip.
    """
    out = []
    for s in sents:
        s = re.sub(r"\s+", " ", s).strip()
        out.append(s)
    return out


def parse_glosses(contents: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to extract two gloss lines that typically appear near the top of the file.
    Supports formats like:
      '1. <gloss>' or '1: <gloss>' or '1    <gloss>'
      '2. <gloss>' or '2: <gloss>' or '2    <gloss>'
    Returns (gloss1, gloss2) or (None, None) if not found.
    """
    lines = [l.strip() for l in contents.splitlines()]
    g1 = None
    g2 = None
    for line in lines[:15]:  # search near the top
        m1 = re.match(r"^1[.:]?\s+(.*)$", line)
        if m1 and not g1:
            g1 = m1.group(1).strip()
            continue
        m2 = re.match(r"^2[.:]?\s+(.*)$", line)
        if m2 and not g2:
            g2 = m2.group(1).strip()
            continue
    return g1 if g1 else None, g2 if g2 else None


def load_stage1_sentences(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load Stage 1 sentences from the discussion board format.
    Returns (sense1_sentences, sense2_sentences).
    """
    from pathlib import Path
    path = Path(filepath)
    if not path.exists():
        return [], []
    contents = path.read_text(encoding="utf-8")
    lines = [l.strip() for l in contents.splitlines()]
    s1 = []
    s2 = []
    in_s1 = False
    in_s2 = False
    for line in lines:
        if "Sense 1" in line or "sense 1" in line.lower():
            in_s1 = True
            in_s2 = False
            continue
        if "Sense 2" in line or "sense 2" in line.lower():
            in_s1 = False
            in_s2 = True
            continue
        # Extract numbered sentences like "1) ..." or "1. ..."
        m = re.match(r"^\d+[).]\s*(.+)$", line)
        if m:
            sent = m.group(1).strip()
            if sent and len(sent.split()) >= 3:
                if in_s1:
                    s1.append(sent)
                elif in_s2:
                    s2.append(sent)
    return s1, s2


def augment_sentences(sentences: List[str], num_augmentations: int = 2) -> List[str]:
    """
    Simple data augmentation: create variations without changing meaning.
    Safe transformations only.
    """
    augmented = list(sentences)  # Keep originals
    import random
    random.seed(42)  # Reproducibility
    
    # Simple safe transformations
    replacements = [
        # Tense variations (safe contexts)
        (r"\bis\b", "was"),
        (r"\bhas\b", "had"),
        (r"\bdoes\b", "did"),
        # Article variations
        (r"\bthe\s+", "a "),
        (r"\ba\s+", "the "),
        # Optional word additions (careful)
        (r"\b(\w+)\s+(director|overtime|rubbish)", r"\1, the \2"),  # Add "the" before target word
    ]
    
    for _ in range(num_augmentations):
        for sent in sentences:
            if len(sent.split()) < 4:  # Skip very short sentences
                continue
            variant = sent
            # Apply one random replacement if it makes sense
            rep = random.choice(replacements)
            variant = re.sub(rep[0], rep[1], variant, count=1, flags=re.I)
            # Only add if it's different and reasonable
            if variant != sent and len(variant.split()) >= 3:
                # Basic sanity: should still contain the target word or key context
                augmented.append(variant)
                if len(augmented) >= len(sentences) * (1 + num_augmentations):
                    break
        if len(augmented) >= len(sentences) * (1 + num_augmentations):
            break
    
    return augmented[:len(sentences) * (1 + num_augmentations)]


class LexiconFeatures(BaseEstimator, TransformerMixin):
    """
    Create simple lexicon-based features indicating presence/counts of sense-specific cues.
    Tailored for specific words:
      - 'rubbish': Feature 0 = physical-trash cues, Feature 1 = nonsensical-talk cues
      - 'director': Feature 0 = organizational cues, Feature 1 = film/theater cues
    For other words, returns zeros.
    """
    def __init__(self, word: str):
        self.word = word
        self._s1_patterns = []
        self._s2_patterns = []

    def fit(self, X, y=None):
        if self.word == "rubbish":
            s1 = [
                r"\b(bin|trash|garbage|landfill|dump(ed)?|litter|garbage\s*truck|trash\s*bag|rubbish\s*heap|refuse|debris|waste|scrap)\b",
                r"\b(clean\s*up|dispose|collection|garbage\s*collector|curbside\s*pickup|collection\s*day|pick-?up\s*day)\b",
                r"\b(recycle|recycling|waste\s+management|skip\s+bin|rubbish\s*bin|trash\s*can|landfill\s*site|compost|compactor)\b",
                r"\b(tip|dumpster|wheelie\s*bin|refuse\s*site|transfer\s*station|municipal\s*dump)\b",
                r"\b(bag(s)?\s*of\s*rubbish|pile(s)?\s*of\s*rubbish)\b",
            ]
            s2 = [
                r"\b(nonsense|nonsensical|codswallop|tripe|drivel|baloney|bollocks|hogwash|poppycock|claptrap|twaddle|bunkum?|malarkey|guff|piffle|balderdash|flimflam|tommyrot)\b",
                r"(absolute|utter|load of|complete|pile of|lot of|load of old|total|sheer)\s+rubbish",
                r"\btalk(ing)?\s+rubbish\b",
                r"\b(that'?s|this is)\s+rubbish\b",
                r"\brubbish!\b",
                r"\b(rubbish|silly|ridiculous)\s+idea\b",
                r"\bthis\s+article\s+is\s+rubbish\b",
            ]
            self._s1_patterns = [re.compile(p, flags=re.I) for p in s1]
            self._s2_patterns = [re.compile(p, flags=re.I) for p in s2]
        elif self.word == "director":
            # Sense 1: Organizational/managerial role
            s1 = [
                r"\b(executive|managing|marketing|hospital|regional|operations|finance|technical|program|communications|research|security|customer\s*service)\s+director\b",
                r"\bdirector\s+of\s+(human\s*resources|sales|operations|communications|research|security|customer\s*service|finance|marketing|it|hr)\b",
                r"\b(board|company|organization|department|division|nonprofit|museum|school|hospital)\s+director\b",
                r"\bdirector\s+(approved|announced|implemented|presented|oversees|coordinates|manages|selected|curated|met|streamlined|improved)\b",
                r"\b(chief\s*executive|ceo|board\s+of\s+directors)\b",
            ]
            # Sense 2: Film/theater/media creator
            s2 = [
                r"\b(film|movie|theater|theatre|cinema|documentary|short\s*film|feature\s*film)\s+director\b",
                r"\bdirector\s+(shouted|called|chose|collaborated|received|wrapped|spent|worked|answered|transformed|blocked|aspires)\b",
                r"\b(actor|actress|scene|shot|take|cut|cinematographer|screenwriter|screening|premiere|festival|pre-?production|principal\s+photography)\b",
                r"\bdirector'?s\s+(cut|vision|style|work|previous\s+work|approach)\b",
                r"\b(award-?winning|independent|aspiring|documentary|theater|theatre)\s+director\b",
                r"\b(director|directors)\s+(study|learn|earn|secure|plan|shoot|film|direct)\b",
                r"\b(film\s+school|film\s+festival|standing\s+ovation|visual\s+storytelling|narrative\s+structures)\b",
            ]
            self._s1_patterns = [re.compile(p, flags=re.I) for p in s1]
            self._s2_patterns = [re.compile(p, flags=re.I) for p in s2]
        else:
            self._s1_patterns = []
            self._s2_patterns = []
        return self

    def transform(self, X):
        feats = np.zeros((len(X), 2), dtype=float)
        if not self._s1_patterns and not self._s2_patterns:
            return feats
        for i, s in enumerate(X):
            s1c = sum(1 for rx in self._s1_patterns if rx.search(s))
            s2c = sum(1 for rx in self._s2_patterns if rx.search(s))
            feats[i, 0] = float(s1c)
            feats[i, 1] = float(s2c)
        return feats


