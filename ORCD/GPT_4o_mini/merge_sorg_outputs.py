import argparse
from pathlib import Path

import pandas as pd


def merge_sorg_outputs(input_dir: Path, output_file: Path) -> Path:
    dataframes = []
    missing_files = []

    for i in range(1, 11):
        file_path = input_dir / f"sorg_output_{i}.csv"
        if not file_path.exists():
            missing_files.append(str(file_path))
            continue
        dataframes.append(pd.read_csv(file_path))

    if missing_files:
        missing_text = "\n".join(missing_files)
        raise FileNotFoundError(
            "Missing one or more input files:\n"
            f"{missing_text}"
        )

    merged_df = pd.concat(dataframes, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sorg_output_1.csv to sorg_output_10.csv into one CSV file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing sorg_output_1.csv ... sorg_output_10.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sorg_output.csv"),
        help="Output merged CSV path",
    )

    args = parser.parse_args()
    output_path = merge_sorg_outputs(args.input_dir, args.output)
    print(f"Merged file saved to: {output_path}")


if __name__ == "__main__":
    main()