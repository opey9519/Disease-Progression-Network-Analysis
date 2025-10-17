# Initial Data Prep
'''
    Purpose:
        Prepare the raw EHR data for network analysis by converting patient diagnosis histories
        into directed, weighted transitions.
'''

import pandas as pd
from collections import Counter
from pathlib import Path

# ---------- Configuration ----------
DATA_DIR = Path(".data")

# ---------- Load Data from CSV files ----------
print("Loading tables")
conditions = pd.read_csv(DATA_DIR / "sampled_condition_occurrence.csv")
persons = pd.read_csv(DATA_DIR / "sampled_person.csv")

print(conditions.columns.tolist())

# Convert date columns to datetime
conditions["condition_start_DATE"] = pd.to_datetime(
    conditions["condition_start_DATE"], errors="coerce")

# DataFrame - merge to bring demographics (currently optional for later slicing)
df = conditions.merge(
    persons[["person_id", "year_of_birth", "gender_concept_id"]], on="person_id", how="left")

# ---------- TRANSITION GENERATION ----------


def generate_transitions(df, pid_col="person_id", diag_col="condition_concept_id",
                         date_col="condition_start_DATE"):
    '''
        Generate sequential (src, dst, weight) pairs per patient.
    '''

    df = df[[pid_col, diag_col, date_col]].dropna()
    df = df.sort_values([pid_col, date_col])
    transitions = Counter()

    for pid, group in df.groupby(pid_col):
        codes = group[diag_col].tolist()
        # Create (a → b) for each consecutive pair
        for a, b in zip(codes, codes[1:]):
            if a != b:
                transitions[(a, b)] += 1

    trans_df = pd.DataFrame([(a, b, w) for (a, b), w in transitions.items()],
                            columns=["src", "dst", "weight"])
    return trans_df


print("Building transitions...")
trans_df = generate_transitions(df)
print(f"→ Generated {len(trans_df)} unique diagnosis transitions.")


# ---------- Saving ----------
out_path = DATA_DIR / "transitions_all.csv"
trans_df.to_csv(out_path, index=False)
print(f"Saved transitions to {out_path}")
