import argparse
import os
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
import re

from wsd_utils import (
    parse_two_sense_file,
    normalize_sentences,
    parse_glosses,
    load_stage1_sentences,
    augment_sentences,
    LexiconFeatures,
)


DATA_FILES = {
    "director": "director.txt",
    "overtime": "overtime.txt",
    "rubbish": "rubbish.txt",
}


def load_word_data(
    base_dir: Path, word: str, use_stage1: bool = True, use_augmentation: bool = True
) -> Tuple[List[str], List[int]]:
    """
    Load training data for a word, optionally including Stage 1 sentences and augmentation.
    """
    path = base_dir / DATA_FILES[word]
    contents = path.read_text(encoding="utf-8")
    s1, s2 = parse_two_sense_file(contents)
    
    # Add Stage 1 sentences if available
    if use_stage1:
        stage1_path = base_dir / f"stage1_{word}_new_sentences.txt"
        if stage1_path.exists():
            stage1_s1, stage1_s2 = load_stage1_sentences(str(stage1_path))
            s1.extend(stage1_s1)
            s2.extend(stage1_s2)
    
    # Augment with glosses as additional training signals
    g1, g2 = parse_glosses(contents)
    if g1:
        s1.append(g1)
    if g2:
        s2.append(g2)
    
    s1 = normalize_sentences(s1)
    s2 = normalize_sentences(s2)
    
    # Apply data augmentation to expand dataset
    if use_augmentation:
        # Augment more aggressively for smaller datasets
        num_aug = 3 if len(s1) + len(s2) < 60 else 2
        s1 = augment_sentences(s1, num_augmentations=num_aug)
        s2 = augment_sentences(s2, num_augmentations=num_aug)
        s1 = normalize_sentences(s1)
        s2 = normalize_sentences(s2)
    
    X = s1 + s2
    # Labels are 1 for first sense, 2 for second sense
    y = [1] * len(s1) + [2] * len(s2)
    return X, y


def build_pipeline(word: str) -> Pipeline:
    # Combine word n-grams and character n-grams for robust small-data performance.
    if word == "rubbish":
        word_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 5),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
        char_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 7),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
    else:
        word_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 4),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
        char_vec = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
        )
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                word_vec,
            ),
            ("lexicon", LexiconFeatures(word=word)),
            (
                "char_tfidf",
                char_vec,
            ),
        ]
    )
    if word == "rubbish":
        # Logistic regression with class weighting; works well with added lexicon features
        clf = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            C=1.0,
        )
    else:
        clf = LinearSVC(C=1.5)
    pipe = Pipeline([("features", features), ("clf", clf)])
    return pipe


def train_and_save_for_word(base_dir: Path, models_dir: Path, word: str, cv: int = 5) -> None:
    X, y = load_word_data(base_dir, word)
    model = build_pipeline(word)
    if cv and len(X) >= cv:
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"{word}: CV accuracy mean={scores.mean():.3f}, std={scores.std():.3f}")
    model.fit(X, y)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"{word}_model.joblib"
    joblib.dump(model, out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved {word} model to {out_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Train WSD models for two-sense words.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Directory containing director.txt, overtime.txt, rubbish.txt",
    )
    parser.add_argument(
        "--models_dir", type=str, default="models", help="Directory to save trained models"
    )
    parser.add_argument(
        "--words",
        type=str,
        nargs="*",
        default=["director", "overtime", "rubbish"],
        help="Subset of words to train (default: all)",
    )
    parser.add_argument("--cv", type=int, default=5, help="CV folds for a quick estimate")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    models_dir = Path(args.models_dir)

    for word in args.words:
        if word not in DATA_FILES:
            print(f"Skipping unknown word: {word}")
            continue
        train_and_save_for_word(base_dir, models_dir, word, cv=args.cv)


if __name__ == "__main__":
    main()


