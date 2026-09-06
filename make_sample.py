"""Build data/applications.csv from the full training CSV.

Takes a stratified sample so the sampled default rate matches the full file,
keeps only the columns the app needs, and writes the result where the app
looks for it.

    python make_sample.py                    # 2000 rows from Loan_default.csv
    python make_sample.py --rows 5000
    python make_sample.py --source path/to/Loan_default.csv
"""
import argparse
import os
import sys

import pandas as pd

MODEL_COLUMNS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio", "Education",
    "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents",
    "LoanPurpose", "HasCoSigner",
]
DEFAULT_SOURCES = ["Loan_default.csv", "data/Loan_default.csv",
                   "data/loan_default.csv", "Loan_Default.csv"]

parser = argparse.ArgumentParser()
parser.add_argument("--source", default=None, help="Path to the full training CSV")
parser.add_argument("--rows", type=int, default=2000, help="Rows to sample")
parser.add_argument("--out", default="data/applications.csv")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

source = args.source or next((p for p in DEFAULT_SOURCES if os.path.exists(p)), None)
if source is None:
    sys.exit("Could not find the training CSV. Looked for:\n  " +
             "\n  ".join(DEFAULT_SOURCES) +
             "\nPass one explicitly:  python make_sample.py --source <path>")

print(f"Reading {source} ...")
frame = pd.read_csv(source)
print(f"  {len(frame):,} rows, {len(frame.columns)} columns")

missing = [c for c in MODEL_COLUMNS if c not in frame.columns]
if missing:
    sys.exit("The source file is missing columns the model needs:\n  " +
             ", ".join(missing))

rows = min(args.rows, len(frame))

if "Default" in frame.columns:
    # Stratified sample: draw from each outcome group in proportion to its
    # size, so the sample carries the same default rate as the full file
    # instead of whatever a lucky draw happens to give.
    #
    # Done by concatenating per-group samples rather than groupby().apply(),
    # because newer pandas drops the grouping column from an apply result.
    share = rows / len(frame)
    parts = []
    for value, group in frame.groupby("Default"):
        take = max(1, round(len(group) * share))
        parts.append(group.sample(min(take, len(group)),
                                  random_state=args.seed))
    sample = pd.concat(parts)

    # Per-group rounding can land a row or two either side of the target, so
    # trim or top up to hit the requested size exactly.
    if len(sample) > rows:
        sample = sample.sample(rows, random_state=args.seed)
    elif len(sample) < rows:
        spare = frame.drop(index=sample.index)
        sample = pd.concat([sample, spare.sample(min(rows - len(sample),
                                                     len(spare)),
                                                 random_state=args.seed)])

    # Shuffle so the two groups are interleaved rather than stacked.
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    full_rate = frame["Default"].mean()
    sample_rate = sample["Default"].mean()
    print(f"  full-file default rate   : {full_rate:.4f}  "
          f"({int(frame['Default'].sum()):,} of {len(frame):,})")
    print(f"  sampled default rate     : {sample_rate:.4f}  "
          f"({int(sample['Default'].sum()):,} of {len(sample):,})")
    print(f"  difference               : {abs(full_rate - sample_rate):.4f}")
else:
    sample = frame.sample(rows, random_state=args.seed).reset_index(drop=True)
    print("  no Default column found; outcomes will show as Unknown")

keep = MODEL_COLUMNS.copy()
if "LoanID" in sample.columns:
    keep = ["LoanID"] + keep
else:
    sample["LoanID"] = [f"L{i:06d}" for i in range(len(sample))]
    keep = ["LoanID"] + keep
if "Default" in sample.columns:
    keep = keep + ["Default"]

sample = sample[keep]

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
sample.to_csv(args.out, index=False)
size_kb = os.path.getsize(args.out) / 1024

print(f"\nWrote {args.out}")
print(f"  {len(sample):,} rows x {len(sample.columns)} columns, {size_kb:,.0f} KB")
print("\nNext:  git add data/applications.csv && git commit -m 'Add application sample'")
