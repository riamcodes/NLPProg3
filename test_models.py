"""
Test script to evaluate WSD models on unseen test sentences.
"""
from cs5322f25prog3 import WSD_Test_director, WSD_Test_overtime, WSD_Test_rubbish
from test_sentences import TEST_SENTENCES


def evaluate_word(word: str, test_func, test_data: dict):
    """
    Evaluate a WSD function on test sentences and report accuracy.
    """
    sense1_sentences = test_data[1]
    sense2_sentences = test_data[2]
    
    # Get predictions
    all_sentences = sense1_sentences + sense2_sentences
    predictions = test_func(all_sentences)
    
    # Expected labels: 20 ones, then 20 twos
    num_per_sense = len(sense1_sentences)
    expected = [1] * num_per_sense + [2] * num_per_sense
    
    # Calculate accuracy
    correct = sum(1 for p, e in zip(predictions, expected) if p == e)
    total = len(expected)
    accuracy = correct / total
    
    # Show detailed results
    print(f"\n{'='*60}")
    print(f"Testing: {word.upper()}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.1%}")
    print()
    
    # Sense 1 results
    sense1_preds = predictions[:num_per_sense]
    sense1_correct = sum(1 for p in sense1_preds if p == 1)
    print(f"Sense 1 (should be 1): {sense1_correct}/{num_per_sense} correct")
    for i, (sent, pred) in enumerate(zip(sense1_sentences, sense1_preds), 1):
        status = "✓" if pred == 1 else "✗"
        print(f"  {status} [{pred}] {sent[:60]}...")
    
    print()
    
    # Sense 2 results
    sense2_preds = predictions[num_per_sense:]
    sense2_correct = sum(1 for p in sense2_preds if p == 2)
    print(f"Sense 2 (should be 2): {sense2_correct}/{num_per_sense} correct")
    for i, (sent, pred) in enumerate(zip(sense2_sentences, sense2_preds), 1):
        status = "✓" if pred == 2 else "✗"
        print(f"  {status} [{pred}] {sent[:60]}...")
    
    return accuracy


def main():
    print("Evaluating WSD Models on Unseen Test Data")
    print("=" * 60)
    
    results = {}
    
    # Test director
    results["director"] = evaluate_word(
        "director", WSD_Test_director, TEST_SENTENCES["director"]
    )
    
    # Test overtime
    results["overtime"] = evaluate_word(
        "overtime", WSD_Test_overtime, TEST_SENTENCES["overtime"]
    )
    
    # Test rubbish
    results["rubbish"] = evaluate_word(
        "rubbish", WSD_Test_rubbish, TEST_SENTENCES["rubbish"]
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for word, acc in results.items():
        print(f"{word:12s}: {acc:.1%} accuracy")
    avg_acc = sum(results.values()) / len(results)
    print(f"{'Average':12s}: {avg_acc:.1%} accuracy")
    print()


if __name__ == "__main__":
    main()

