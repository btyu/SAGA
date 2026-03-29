#!/usr/bin/env python3
import pandas as pd
import argparse
import math
from pathlib import Path


def split_csv(input_file, output_prefix, num_chunks):
    """Split CSV into chunks for parallel processing."""
    df = pd.read_csv(input_file)
    total_rows = len(df)
    chunk_size = math.ceil(total_rows / num_chunks)

    print(
        f"Splitting {total_rows} rows into {num_chunks} chunks of ~{chunk_size} rows each"
    )

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)

        if start_idx >= total_rows:
            break

        chunk_df = df.iloc[start_idx:end_idx]
        output_file = f"{output_prefix}_chunk_{i:03d}.csv"
        chunk_df.to_csv(output_file, index=False)
        print(f"Created {output_file} with {len(chunk_df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split CSV for parallel processing")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("--chunks",
                        type=int,
                        default=10,
                        help="Number of chunks")
    parser.add_argument("--prefix", default="input", help="Output file prefix")

    args = parser.parse_args()
    split_csv(args.input, args.prefix, args.chunks)


