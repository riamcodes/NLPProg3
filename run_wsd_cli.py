import argparse
from pathlib import Path

from cs5322f25prog3 import (
    WSD_Test_director,
    WSD_Test_overtime,
    WSD_Test_rubbish,
)


def main():
    parser = argparse.ArgumentParser(description="Run WSD on a test file and save results.")
    parser.add_argument("--word", required=True, choices=["director", "overtime", "rubbish"])
    parser.add_argument("--input", required=True, help="Path to <word>_test.txt with 50 lines")
    parser.add_argument("--output", required=True, help="Path to write result_<word>_*.txt")
    args = parser.parse_args()

    input_path = Path(args.input)
    lines = [l.rstrip("\n") for l in input_path.read_text(encoding="utf-8").splitlines()]
    if args.word == "director":
        preds = WSD_Test_director(lines)
    elif args.word == "overtime":
        preds = WSD_Test_overtime(lines)
    else:
        preds = WSD_Test_rubbish(lines)

    out_path = Path(args.output)
    out_path.write_text("\n".join(str(p) for p in preds) + "\n", encoding="utf-8")
    print(f"Wrote {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()


