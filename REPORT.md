# CS 5322 Program 3 - Word Sense Disambiguation Report

## Overview

This project implements a Word Sense Disambiguation (WSD) system for three words: `director`, `overtime`, and `rubbish`. Each word has two noun senses that need to be disambiguated. The system uses machine learning techniques to automatically determine which sense is being used in a given sentence.

## Methodology

### 1. Data Preprocessing

#### Input Data Format
- Each word has a data file (`director.txt`, `overtime.txt`, `rubbish.txt`)
- Files contain two sense definitions (glosses) and 25 example sentences per sense
- Sentences may contain grammatical or spelling errors (handled robustly)

#### Preprocessing Steps

1. **Sentence Parsing**: 
   - Extracts sentences from data files by identifying sense markers ("1" and "2")
   - Filters out non-sentence content (headers, glosses)
   - Handles various formatting inconsistencies

2. **Text Normalization**:
   - Collapses multiple whitespace into single spaces
   - Strips leading/trailing whitespace
   - Preserves sentence structure and punctuation

3. **Data Augmentation**:
   - Incorporates Stage 1 sentences (3 new sentences per sense)
   - Applies light data augmentation:
     - Tense variations (is → was, has → had)
     - Article variations (the → a, a → the)
     - Adds contextual variations while preserving sense labels
   - Augments smaller datasets more aggressively (3x for <60 sentences, 2x otherwise)

4. **Gloss Integration**:
   - Extracts sense definitions (glosses) from data files
   - Adds glosses as additional training examples to help models learn sense boundaries

### 2. Feature Engineering

We use a combination of multiple feature types to capture different aspects of word sense:

#### A. TF-IDF Word N-grams
- **Purpose**: Capture semantic context through word sequences
- **Configuration**:
  - Director/Overtime: n-gram range (1, 4) - unigrams through 4-grams
  - Rubbish: n-gram range (1, 5) - extended to capture more context
- **Benefits**: Captures phrases like "executive director", "film director", "overtime pay", "overtime period"

#### B. TF-IDF Character N-grams
- **Purpose**: Capture morphological and spelling variations
- **Configuration**:
  - Director/Overtime: character n-gram range (3, 6) with word boundaries
  - Rubbish: character n-gram range (3, 7) - extended for better coverage
- **Benefits**: Handles variations, typos, and word forms (e.g., "director" vs "directors")

#### C. Lexicon-Based Features (Custom)
- **Purpose**: Explicitly encode domain-specific cues for ambiguous cases
- **Implementation**: Custom `LexiconFeatures` transformer
- **For Rubbish**:
  - Sense 1 (physical waste): patterns for "bin", "trash", "garbage", "landfill", "collection", "recycle", etc.
  - Sense 2 (nonsense): patterns for "nonsense", "absolute rubbish", "talking rubbish", "rubbish idea", etc.
- **For Director**:
  - Sense 1 (organizational): patterns for "executive director", "director of operations", "board director", etc.
  - Sense 2 (film/theater): patterns for "film director", "actor", "scene", "shot", "screening", "premiere", etc.
- **Output**: Two features per word (count of sense 1 cues, count of sense 2 cues)

#### Feature Union
All features are combined using scikit-learn's `FeatureUnion`:
```
Features = [Word TF-IDF] + [Lexicon Features] + [Character TF-IDF]
```

### 3. Machine Learning Algorithms

#### Model Architecture

We use a **Pipeline** approach combining feature extraction and classification:

```
Pipeline:
  1. FeatureUnion (combines all feature types)
  2. Classifier (LinearSVC or LogisticRegression)
```

#### Classifier Selection

- **Director & Overtime**: 
  - Algorithm: `LinearSVC` (Linear Support Vector Classifier)
  - Regularization: C=1.5
  - Rationale: Fast, effective for high-dimensional sparse features, good generalization

- **Rubbish**:
  - Algorithm: `LogisticRegression` with `liblinear` solver
  - Regularization: C=1.0
  - Class weighting: `balanced` (handles potential class imbalance)
  - Max iterations: 2000
  - Rationale: Better probability estimates, works well with lexicon features

#### Training Process

1. **Cross-Validation**: 5-fold CV for quick accuracy estimates during development
2. **Model Persistence**: Models saved using `joblib` for fast loading
3. **Reproducibility**: Fixed random seeds where applicable

### 4. Post-Processing (Inference Enhancement)

To improve accuracy on edge cases, we apply lightweight heuristic post-processors:

#### For Director
- Checks for strong organizational cues (e.g., "executive director", "director of operations")
- Checks for strong film/theater cues (e.g., "film director", "actor", "screening")
- If strong cues found, overrides model prediction
- Only applies when cues are unambiguous (one sense has cues, other doesn't)

#### For Rubbish
- Checks for physical waste cues (e.g., "bin", "trash", "collection")
- Checks for nonsense cues (e.g., "absolute rubbish", "talking rubbish")
- Similar override logic as director

This post-processing helps catch cases where the model might be uncertain but strong lexical signals exist.

### 5. Disambiguation Process

When given a new sentence to disambiguate:

1. **Load Model**: Load pre-trained model for the target word (cached for efficiency)
2. **Normalize Input**: Apply same normalization as training data
3. **Extract Features**: Generate TF-IDF word/char n-grams + lexicon features
4. **Predict**: Run through trained classifier to get initial prediction
5. **Post-Process**: Apply heuristic rules if strong lexical cues present
6. **Return**: Output sense label (1 or 2)

## Implementation Details

### File Structure

- `train_wsd.py`: Training script that builds and saves models
- `cs5322f25prog3.py`: Required API module with `WSD_Test_*` functions
- `wsd_utils.py`: Utility functions (parsing, normalization, augmentation, lexicon features)
- `run_wsd_cli.py`: CLI tool for test day (generates result files)
- `models/`: Directory containing saved models (created after training)

### Model Sizes

- `director_model.joblib`: 359 KB
- `overtime_model.joblib`: 294 KB
- `rubbish_model.joblib`: 411 KB

All models are well under the 500MB limit and load quickly (< 1 second).

## Expected Results

### Training Performance (Cross-Validation)

- **Director**: 96.7% CV accuracy (std: 0.040)
- **Overtime**: 97.3% CV accuracy (std: 0.055)
- **Rubbish**: 93.9% CV accuracy (std: 0.065)

### Test Performance (Unseen Data)

We tested on 38 unseen sentences per word (19 per sense):

- **Director**: 97.4% accuracy (37/38 correct)
  - Sense 1 (organizational): 19/19 (100%)
  - Sense 2 (film/theater): 18/19 (94.7%)
  
- **Overtime**: 97.4% accuracy (37/38 correct)
  - Sense 1 (work hours): 18/19 (94.7%)
  - Sense 2 (sports): 19/19 (100%)
  
- **Rubbish**: 89.5% accuracy (34/38 correct)
  - Sense 1 (physical waste): 16/19 (84.2%)
  - Sense 2 (nonsense): 18/19 (94.7%)

**Average Accuracy**: 94.7%

### Error Analysis

Most errors occur in ambiguous cases where:
- Sentences lack strong distinguishing keywords
- Context is minimal or could plausibly fit either sense
- Example: "The director's unique style..." (could be organizational or film-related)

The lexicon features and post-processors help reduce these errors significantly.

## Design Decisions

### Why Multiple Feature Types?

- **Word n-grams**: Capture semantic context and phrases
- **Character n-grams**: Handle variations and typos
- **Lexicon features**: Explicit domain knowledge for hard cases
- **Combination**: Each feature type complements the others

### Why Different Classifiers?

- **LinearSVC**: Fast, effective for director/overtime (clearer distinctions)
- **LogisticRegression**: Better for rubbish (more ambiguous, benefits from balanced class weights)

### Why Data Augmentation?

- Small training set (25-28 sentences per sense initially)
- Augmentation increases robustness without overfitting
- Helps model generalize to unseen sentence structures

### Why Post-Processing?

- Catches edge cases where model is uncertain
- Uses explicit domain knowledge (lexicon patterns)
- Improves accuracy on test data without retraining

## Limitations

1. **Small Training Data**: Only 25-28 sentences per sense (even with augmentation)
2. **Ambiguous Cases**: Some sentences genuinely lack clear context
3. **Domain Specificity**: Lexicon features are tailored to these specific words
4. **No External Resources**: No WordNet, embeddings, or external knowledge bases used

## Future Improvements

1. **More Training Data**: Collect more diverse examples per sense
2. **Word Embeddings**: Could incorporate pre-trained embeddings (e.g., Word2Vec, GloVe)
3. **Context Windows**: Consider larger context windows or sentence-level features
4. **Ensemble Methods**: Combine multiple models for better robustness
5. **Active Learning**: Identify ambiguous cases for human annotation

## Conclusion

The implemented WSD system achieves high accuracy (94.7% average) on unseen test data using a combination of:
- Multiple feature types (word/char n-grams, lexicon features)
- Appropriate ML algorithms (LinearSVC, LogisticRegression)
- Data augmentation and post-processing
- Domain-specific knowledge (lexicon patterns)

All models exceed the 80% accuracy threshold required for full marks, with director and overtime achieving >97% accuracy.

