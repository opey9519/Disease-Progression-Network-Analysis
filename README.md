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


### Step A – Data Preprocessing (Completed)

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
