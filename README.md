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

This step converts patient diagnosis transitions into a directed, weighted network for analysis and ML.

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
