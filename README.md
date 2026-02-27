# Feature Importance Stability Analysis on CICIDS2017

This repository contains the source code for a research project investigating the most important features for machine learning algorithms in Network Intrusion Detection Systems (NIDS).

## Research Question

The main objective of this study is to answer the following question:

> **"Are the most important features different depending on which attack type occurs?"**

To address this, we implemented a sliding window analysis that evaluates feature importance over time using multiple feature selection methods:

- Random Forest (Gini Importance)
- Permutation Importance
- ANOVA F-value
- Mutual Information

By aggregating these scores, the project builds a consensus signature for each attack type and evaluates their similarities.

## Project Structure

- `src/`: Contains the Python analysis scripts.
  - `main.py`: The core script that computes feature importance across the 4 methods, generates heatmaps, and calculates the Jaccard similarity matrix between attack signatures.
  - `attacks_count_per_window.py`: A utility script to analyze the dataset distribution, computing the percentage and count of each attack type per sliding window.
- `data/`: Directory for the dataset. (Note: The raw CSV is too large to be hosted here).
  - `Genarate_Dataset_on_kaggle.ipynb`: A Jupyter Notebook designed to run on Kaggle to clean, merge, and filter the raw CICIDS2017 dataset. **Please take the time to read his instruction at the beginning of it in order to get the dataset (`cicids2017_filtered.csv`)!**
- `results/`: Directory where output plots (`png/`) and metrics (`csv/`) are generated.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YouGoooo/cicids2017-feature-analysis-per-attack-types.git
   cd cicids2017-feature-analysis-per-attack-types
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Once the dataset is in place, you can run the analysis scripts from the root directory.

**1. Analyze Attack Distribution:**
To generate the window-by-window attack counts and percentages:

```bash
python src/attacks_count_per_window.py
```

**2. Run the Main Feature Analysis:**
To compute the feature importances, generate all heatmaps, and extract the consensus signatures:

```bash
python src/main.py
```

All generated figures and matrices will be saved in the `results/` folder.

## Technologies

- Python 3.x
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- **Rodrigo Sanches Miani** (Internship Supervisor) and **Elaine Fari** (PhD Candidate) for their guidance and support during this internship at the **Federal University of Uberlândia (UFU)**.
- **Eric Anacleto Ribeiro** for the extensive work on CICIDS2017 preprocessing and cleaning.
- The original CICIDS2017 dataset creators (Canadian Institute for Cybersecurity).
