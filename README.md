# Higgs Boson Detection using Evolutionary and Ensemble Learning

## Project Overview
This project implements and combines Evolutionary Learning and Ensemble Learning for binary classification, demonstrated on Higgs Boson detection. We successfully built and tested the core components.

## What We Actually Built

### ✅ Working Components:
1. **Data Preprocessing** - Synthetic data generation and preprocessing pipeline
2. **Evolutionary Learning (GA)** - Genetic Programming classifier using DEAP
3. **Ensemble Learning (EL)** - Voting ensemble from evolutionary population
4. **Comparative Analysis** - Performance comparison between methods

### 🔄 Ready for Implementation:
- **Active Learning** - Code structure ready (requires modAL installation)

## Project Structure
higgs_boson_project/
├── src/
│ ├── data_preprocessing.py # Data loading and preprocessing
│ ├── evolutionary_learning.py # Genetic Programming implementation
│ ├── ensemble_learning.py # Ensemble voting methods
│ └── active_learning.py # Active learning structure (needs modAL)
├── main.py # Main experiment runner
└── requirements.txt # Dependencies


## Installation & Setup
```bash
# Install required packages
pip install numpy pandas scikit-learn deap matplotlib seaborn

# Run the project
python main.py

What Actually Runs
When you run python main.py:

Automatically generates synthetic data (since real Higgs dataset not required)
Runs Evolutionary Learning (GA) with 30 individuals for 10 generations
Creates Ensemble (GA+EL) from the final population
Compares performance between GA and GA+EL approaches
