import pandas as pd
from pathlib import Path
from data_prep import generate_transitions
from build_graphs import build_directed_graph, graph_to_pyg
import torch
import networkx as nx

# Configuration
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output/demographic_networks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = 2025
AGE_BINS = [0, 17, 49, 64, 150]
AGE_LABELS = ["<18", "18–49", "50–64", "65+"]


def prepare_demographics():
    """Load condition and person data, attach demographics."""
    print("Loading tables for demographic slicing...")
    conditions = pd.read_csv(DATA_DIR / "sampled_condition_occurrence.csv")
    persons = pd.read_csv(DATA_DIR / "sampled_person.csv")

    # Ensure correct column types
    conditions["condition_start_DATE"] = pd.to_datetime(
        conditions["condition_start_DATE"], errors="coerce")

    persons["age"] = CURRENT_YEAR - persons["year_of_birth"]
    persons["age_group"] = pd.cut(persons["age"], bins=AGE_BINS,
                                  labels=AGE_LABELS, right=True)

    # Map OMOP gender concept IDs
    gender_map = {
        8507: "Male",   # OMOP standard: 8507 = male
        8532: "Female"  # OMOP standard: 8532 = female
    }
    persons["gender"] = persons["gender_concept_id"].map(
        gender_map).fillna("Other/Unknown")

    return conditions, persons


def build_group_networks():
    """Generate transitions and build networks per demographic group."""
    conditions, persons = prepare_demographics()

    for gender in persons["gender"].unique():
        for age_group in persons["age_group"].dropna().unique():
            group_name = f"{gender}_{age_group}"
            print(f"\n🧩 Building network for group: {group_name}")

            # Select patient subset
            subset_ids = persons[
                (persons["gender"] == gender) &
                (persons["age_group"] == age_group)
            ]["person_id"]

            subset_conditions = conditions[conditions["person_id"].isin(
                subset_ids)]

            if subset_conditions.empty:
                print(f"⚠️ No data for {group_name}, skipping.")
                continue

            # Generate transitions using Step A logic
            trans_df = generate_transitions(subset_conditions)

            # Save transitions
            trans_path = OUTPUT_DIR / f"transitions_{group_name}.csv"
            trans_df.to_csv(trans_path, index=False)
            print(f"Saved transitions: {trans_path}")

            # Build graph (Step B)
            G = build_directed_graph(trans_df, min_weight=3)

            # Save outputs
            prefix = OUTPUT_DIR / f"network_{group_name}"
            nx.write_graphml(G, f"{prefix}.graphml")
            torch.save(graph_to_pyg(G), f"{prefix}.pt")
            print(f"Saved network: {prefix}.graphml / .pt")


if __name__ == "__main__":
    build_group_networks()
