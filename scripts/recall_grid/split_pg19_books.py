#!/usr/bin/env python3
"""split_pg19_books.py - write one .txt per book from PG-19 parquet shards (filler books for gen_mixup_64k.py).

The mix64k run used one train shard of `emozilla/pg19` (`data/train-00000-of-00023-5263dc4323e881d2.parquet`,
sha256 589252c8a3e870278db2b5d3f0da73e4845085e852cb024056c37b76d088ba41): 1,244 rows, 1,185 files, because a book
whose short_book_title repeats overwrites the earlier file. Needs pyarrow.

  python scripts/recall_grid/split_pg19_books.py --out BOOKS_DIR train-00000-of-00023-5263dc4323e881d2.parquet
"""
import argparse, os

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("parquet", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for path in a.parquet:
        for r in pq.read_table(path).to_pylist():
            # The train shards have no book_id column, so every name ends in "__None"; kept so the files match the run.
            bid = f"{r.get('short_book_title')}__{r.get('book_id')}"
            name = "".join(c if c.isalnum() else "_" for c in bid)[:80] + ".txt"
            with open(os.path.join(a.out, name), "w") as f:
                f.write(r["text"])
            n += 1
    print(f"{n} books, {len(os.listdir(a.out))} files in {a.out}")


if __name__ == "__main__":
    main()
