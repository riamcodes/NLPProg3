# CS 5322 F25 - Program 3: Word Sense Disambiguation

## 📋 Quick Start Guide

This project implements a Word Sense Disambiguation (WSD) system for three words: `director`, `overtime`, and `rubbish`. Each word has two senses that the system can automatically identify.

**For detailed methodology and technical information, see [REPORT.md](REPORT.md)**

## 🚀 Setup (One-Time)

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Train the Models

**Important**: You must train the models before using the WSD functions!

```bash
python train_wsd.py --base_dir "." --models_dir "models"
```

This will:
- Parse the training data from `director.txt`, `overtime.txt`, `rubbish.txt`
- Include Stage 1 sentences (if present)
- Apply data augmentation
- Train three separate models (one per word)
- Save models to `models/` directory

**Training time**: ~10-30 seconds  
**Model sizes**: All under 500KB (well within limits)

## 📖 Usage

### Option 1: Using the Python API (For Testing/Development)

```python
from cs5322f25prog3 import WSD_Test_director, WSD_Test_overtime, WSD_Test_rubbish

# Test director
sentences = [
    "The managing director approved the budget.",
    "The film director shouted 'cut' after the scene."
]
results = WSD_Test_director(sentences)
print(results)  # Output: [1, 2]

# Test overtime
sentences = [
    "Employees worked overtime last week.",
    "The game went into overtime."
]
results = WSD_Test_overtime(sentences)
print(results)  # Output: [1, 2]

# Test rubbish
sentences = [
    "Please take out the rubbish.",
    "That's absolute rubbish!"
]
results = WSD_Test_rubbish(sentences)
print(results)  # Output: [1, 2]
```

**Note**: Each function returns a list of integers (1 or 2), where:
- `1` = First sense (organizational director, work hours overtime, physical waste rubbish)
- `2` = Second sense (film director, sports overtime, nonsense rubbish)

### Option 2: Using the CLI (For Test Day)

On test day (12/9 at 8:30pm), you'll download three test files from Canvas:
- `director_test.txt` (50 sentences)
- `overtime_test.txt` (50 sentences)
- `rubbish_test.txt` (50 sentences)

Generate the required result files:

```bash
# Replace "First_Last" with one group member's name
python run_wsd_cli.py --word director --input director_test.txt --output "result_director_First_Last.txt"
python run_wsd_cli.py --word overtime --input overtime_test.txt --output "result_overtime_First_Last.txt"
python run_wsd_cli.py --word rubbish --input rubbish_test.txt --output "result_rubbish_First_Last.txt"
```

Each result file will contain exactly 50 lines, each line being `1` or `2`.

**Time limit**: 10 minutes total for all 3 files (not 10 minutes each)

## 📁 Project Structure

```
.
├── README.md                          # This file
├── REPORT.md                          # Detailed methodology and technical report
├── requirements.txt                   # Python dependencies
├── train_wsd.py                       # Training script (run this first!)
├── cs5322f25prog3.py                 # Required API module (grading entry point)
├── run_wsd_cli.py                    # CLI tool for test day
├── wsd_utils.py                       # Utility functions
├── director.txt                       # Training data for "director"
├── overtime.txt                       # Training data for "overtime"
├── rubbish.txt                        # Training data for "rubbish"
├── stage1_director_new_sentences.txt  # Stage 1 sentences (3 per sense)
├── stage1_overtime_new_sentences.txt  # Stage 1 sentences (3 per sense)
├── stage1_rubbish_new_sentences.txt   # Stage 1 sentences (3 per sense)
└── models/                            # Saved models (created after training)
    ├── director_model.joblib
    ├── overtime_model.joblib
    └── rubbish_model.joblib
```

## 🧪 Testing the System

### Quick Test

```python
from cs5322f25prog3 import WSD_Test_director
result = WSD_Test_director(["The executive director met with the board."])
print(result)  # Should output: [1]
```

### Run Full Test Suite

We've included test scripts to verify everything works:

```bash
# Test on unseen data (38 sentences per word)
python test_models.py
```

This will show detailed accuracy results for each word.

## 📊 Expected Performance

Based on testing with unseen data:

- **Director**: ~97% accuracy
- **Overtime**: ~97% accuracy  
- **Rubbish**: ~90% accuracy
- **Average**: ~95% accuracy

All models exceed the 80% threshold required for full marks.

## 🔧 Troubleshooting

### "Model file not found" Error

**Solution**: Run the training script first:
```bash
python train_wsd.py --base_dir "." --models_dir "models"
```

### "ModuleNotFoundError" Error

**Solution**: Make sure you've activated the virtual environment and installed dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Models Not Loading

**Solution**: Ensure models directory exists and contains `.joblib` files:
```bash
ls models/
# Should show: director_model.joblib, overtime_model.joblib, rubbish_model.joblib
```

## 📝 What to Submit

By **2:00pm on 12/9 (Tuesday)**:

1. ✅ Training script: `train_wsd.py`
2. ✅ Saved models: `models/*.joblib` (all three files)
3. ✅ Required module: `cs5322f25prog3.py`
4. ✅ **Report**: `REPORT.md` (see below)

### Test Day Submission (8:30pm on 12/9)

After downloading test files from Canvas, generate and upload:
- `result_director_<First_Last>.txt`
- `result_overtime_<First_Last>.txt`
- `result_rubbish_<First_Last>.txt`

**Deadline**: 8:40pm (10 minutes after test files are released)

## 📚 Documentation

- **[REPORT.md](REPORT.md)**: Complete technical report including:
  - Methodology and preprocessing steps
  - Feature engineering details
  - Machine learning algorithms used
  - Disambiguation process
  - Expected results and performance analysis
  - Design decisions and limitations

## 👥 For Group Partners

### First Time Setup

1. Clone/download this repository
2. Follow the "Setup" section above
3. Run `python train_wsd.py` to generate models
4. Test with the examples in "Usage" section

### Before Test Day

1. **Verify models are trained**: Check that `models/` directory has 3 `.joblib` files
2. **Test the CLI**: Run `python run_wsd_cli.py --help` to ensure it works
3. **Practice**: Use the test scripts to verify everything works
4. **Read the report**: Understand how the system works (see REPORT.md)

### On Test Day

1. Download test files from Canvas at 8:30pm
2. Run the CLI commands (see "Option 2" above)
3. Verify output files have exactly 50 lines each
4. Upload all 3 result files before 8:40pm

## 🎯 Key Features

- ✅ Fast prediction (< 1 second for 50 sentences)
- ✅ Small model sizes (< 500KB each)
- ✅ High accuracy (>90% on all words)
- ✅ Robust to spelling/grammar errors
- ✅ No external dependencies (only scikit-learn)
- ✅ Built from scratch (no existing WSD code)

## 📞 Questions?

- Check [REPORT.md](REPORT.md) for technical details
- Review the code comments in each file
- Test with the provided test scripts

---

**Good luck on test day! 🚀**
