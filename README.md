# Disease Progression Network Analysis

## Project Overview

This repository contains code and data pipelines for analyzing disease progression pathways using electronic health records (EHR).
The goal is to investigate how illnesses develop over time and whether progression differs by age or gender.

We model the data as a temporal, directed network:

- **Nodes:** diagnoses (optionally medications or labs in the future)

- **Edges:** transitions from one diagnosis to the next

- **Edge weight:** number of patients with that transition

## Repository Structure
...  

## Project
### Environment Setup
Create virtual environment
```
python -m venv .venv
```
Activate virtual environment
```
source .venv/bin/activate
```
Install dependencies from requirements.txt
```
pip install -r requirements.txt
```


### Step A – Data Preprocessing

**Purpose:** Convert raw EHR diagnosis data into a directed, weighted transition network.

**Inputs:**
- data/sample_condition_occurrence.csv
- data/sample_person.csv

**Output:**
- data/transitions_all.csv

**Columns:** src, dst, weight

**Each row = one directed transition with the number of patients who experienced it**

**How it works:**
1. Load diagnoses and demographics
2. Convert diagnosis dates to datetime
3. Sort each patient’s diagnoses by date
4. Generate consecutive diagnosis pairs (src → dst)
5. Count the number of patients for each pair
6. Save the result as transitions_all.csv

### Step B – Build Network

**Purpose:** This step converts patient diagnosis transitions into a directed, weighted network for analysis and ML.

**Input:**  
- CSV file with columns: `src`, `dst`, `weight` (generated in Step A / `data_prep.py`)

**Process:**  
1. Filter edges with weight below a threshold (`--min_weight`) to reduce noise.  
2. Build a directed graph using NetworkX.  
3. Convert the NetworkX graph into a PyTorch Geometric `Data` object.

**Output:**  
- `output_prefix.graphml` → GraphML file for visualization (Gephi, Cytoscape, etc.)  
- `output_prefix.pt` → PyG Data object for downstream ML models

**Usage:**

```bash
python3 src/build_graphs.py .data/transitions_all.csv output/network_stageB --min_weight 1
```

### Step C – Demographic slicing
**Purpose:** This step generates separate disease progression networks for demographic subgroups based on patient age and gender.

Using the same pipeline as Steps A and B, src/demographic_slicing.py:
Loads OMOP-style tables:
- data/sampled_person.csv
- data/sampled_condition_occurrence.csv

**Derives demographic attributes:**
- Age = CURRENT_YEAR - year_of_birth
- Age Groups = <18, 18–49, 50–64, 65+
- Gender = mapped from OMOP gender_concept_id (8507 = Male, 8532 = Female)

**Loops through all age–gender combinations and builds networks:**
- Runs Step A → generate_transitions
- Runs Step B → build_directed_graph, graph_to_pyg

**Saves outputs for each group under:**
```
output/demographic_networks/
```

### Step D – Network Analysis

**Purpose:** This step analyzes the demographic subgroup networks (from Step C) to identify shared and unique disease progression patterns.

**src/analysis.py performs:**
- High-frequency transitions
- Extracts top-weighted edges per group (most common progressions).

**Centrality metrics**
- PageRank
- in-degree
- out-degree identify “hub” diagnoses.

**Path analysis**
Finds frequent 2–3 step progression pathways (e.g., diabetes → kidney disease → dialysis).

**Community detection**
Clusters diagnoses into communities using greedy modularity.

**Output Directory**
All results are saved under:
```
output/analysis_results/
```

**Each file corresponds to a demographic group:**
| File                          | Description                 |
| ----------------------------- | --------------------------- |
| `top_transitions_<group>.csv` | Most frequent transitions   |
| `centrality_<group>.csv`      | Node centrality measures    |
| `paths_<group>.csv`           | Common short pathways       |
| `communities_<group>.csv`     | Detected diagnosis clusters |
| `summary_all_groups.csv`      | Overview across all groups  | 

**Run Command**
```
cd src
python3 analysis.py
```
