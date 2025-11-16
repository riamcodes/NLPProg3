# Compliance Checklist - CS 5322 Program 3

## ✅ Stage 1: Beefing up the data set (10 points)
- [x] Created 3 new sentences for sense 1 of director
- [x] Created 3 new sentences for sense 2 of director  
- [x] Sentences are sufficiently different from training data
- [x] File: `stage1_director_new_sentences.txt`
- [x] Also created for overtime and rubbish (bonus)

## ✅ Stage 2: Building WSD tool (90 points)

### Requirements Met:

#### Core Requirements:
- [x] Built WSD tool in Python
- [x] Can determine which of two senses is used in a sentence
- [x] Built one tool for each word separately (3 separate models)
- [x] Used machine learning method (TF-IDF + LinearSVC/LogisticRegression)
- [x] Used text preprocessing (normalization, n-grams, character n-grams)
- [x] **Did NOT download existing WSD code** - built from scratch using only scikit-learn (general ML library, not WSD-specific)
- [x] Models can be saved and loaded without retraining
- [x] Model sizes are reasonable:
  - director_model.joblib: 359KB (< 500MB ✅)
  - overtime_model.joblib: 294KB (< 500MB ✅)
  - rubbish_model.joblib: 411KB (< 500MB ✅)
- [x] Loading time is reasonable (models load in < 1 second)
- [x] Prediction is fast after loading (< 1 second for 50 sentences)

#### Required Module: `cs5322f25prog3.py`
- [x] Module exists: `cs5322f25prog3.py`
- [x] `WSD_Test_director(list)` function exists
  - [x] Takes list of strings (sentences containing "director")
  - [x] Returns list of numbers (1 or 2, NOT 0 or 1)
  - [x] Order matches input sentences
- [x] `WSD_Test_overtime(list)` function exists
  - [x] Takes list of strings (sentences containing "overtime")
  - [x] Returns list of numbers (1 or 2, NOT 0 or 1)
  - [x] Order matches input sentences
- [x] `WSD_Test_rubbish(list)` function exists
  - [x] Takes list of strings (sentences containing "rubbish")
  - [x] Returns list of numbers (1 or 2, NOT 0 or 1)
  - [x] Order matches input sentences
- [x] Models are loaded inside each function (as required)

## ✅ What to Hand In:

- [x] Training program: `train_wsd.py`
- [x] Saved models: `models/director_model.joblib`, `models/overtime_model.joblib`, `models/rubbish_model.joblib`
- [x] Required module: `cs5322f25prog3.py`
- [ ] **Report** (needs to be written - see below)

## ✅ Testing Procedure Readiness:

- [x] CLI tool exists: `run_wsd_cli.py`
- [x] Can read `<word>_test.txt` files (50 sentences)
- [x] Can write `result_<word>_<First_Last>.txt` files
- [x] Output format: 50 lines, each line is 1 or 2
- [x] Order is preserved correctly

## ✅ Model Performance:

All models tested on unseen data (38 sentences per word):
- **Director**: 97.4% accuracy (37/38) ✅
- **Overtime**: 97.4% accuracy (37/38) ✅
- **Rubbish**: 89.5% accuracy (34/38) ✅
- **Average**: 94.7% accuracy

All models exceed 80% accuracy threshold for full marks.

## ⚠️ Missing Item:

- [ ] **Report** detailing:
  - Preprocessing steps
  - Machine learning algorithm used
  - Steps to disambiguate a sentence
  - This needs to be written before submission

## Summary:

✅ **All technical requirements are met.** The only missing piece is the written report documenting the methodology.

