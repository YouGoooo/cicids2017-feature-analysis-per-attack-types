# Feature Importance Stability Analysis on CICIDS2017

This repository contains the source code for a research project investigating what are the most important fetaures for machine learning algorithmes in Network Intrusion Detection Systems (NIDS).

## Research Question

The main objective of this study is to answer the following question:

> **"Are the most important features different depending on which attack type occurs?"**

To address this, we implemented a **sliding window analysis** that evaluates feature importance dynamically over time using multiple feature selection algorithms (Random Forest, SelectKBest, Permutation Importance, and Mutual Information).

## Project Structure

- `src/`: Contains the main Python analysis script (`analyze_features_per_windows.py`).
- `data/`: Placeholder folder for the dataset (see instructions below).
- `results/`: Directory where output plots and CSV metrics will be generated.

## Dataset & Preprocessing

This project uses the **CICIDS2017** dataset. Instead of using the raw PCAP or CSV files, we rely on the high-quality preprocessed version established by **Eric Anacleto Ribeiro**.

We highly recommend reviewing his comprehensive preprocessing pipeline to understand how the features were cleaned and engineered:

- **Preprocessing details:** [GitHub - ml-intrusion-detection-cicids2017](https://github.com/anacletu/ml-intrusion-detection-cicids2017/blob/9b4f49f3234027e95660adb497a94eed9636f44c/cicids2017-comprehensive-data-processing-for-ml.ipynb)
- **Model Comparison:** [CICIDS2017 - ML Models Comparison: Supervised](https://github.com/anacletu/ml-intrusion-detection-cicids2017)

### How to get the data

Due to GitHub's file size limits, the dataset is not included in this repository. Please follow these steps to reproduce our environment:

1.  **Download:** Go to the [Kaggle Output section](https://www.kaggle.com/code/ericanacletoribeiro/cicids2017-comprehensive-data-processing-for-ml/output) of the preprocessing notebook.
2.  **Filter:** Personally I choose to remove "Monday" data from the dataset because it only contains Benign traffic but you can totally run my code while kepping Monday's data.
3.  **Install:** Rename your file to `cicids2017_filtered.csv` and place it in the `data/` folder of this repository.

> **Note:** The script expects the path: `./data/cicids2017_filtered.csv`

## Installation

1.  Clone this repository:

    ```bash
    git clone [https://github.com/YOUR_USERNAME/cicids2017-feature-analysis-per-attack-types.git](https://github.com/YOUR_USERNAME/cicids2017-feature-analysis-per-attack-types.git)
    cd cicids2017-feature-analysis-per-attack-types
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Ensure your dataset is placed in the `data/` folder, then run the main analysis script:

```bash
python src/analyze_features_per_windows.py
```

The script will:

1.  Load the dataset.
2.  Iterate through the data using sliding windows.
3.  Compute feature importance scores using RF, SelectKBest, Permutation, and Mutual Info.
4.  Save the resulting plots and CSV reports in the `results/` folder.

## Technologies

- Python 3.x
- Scikit-learn
- Pandas / NumPy
- Matplotlib

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- **Rodrigo Sanches Miani** (Internship Supervisor) and **Elaine Fari** (PhD Candidate) for their guidance and support during this internship at the **Federal University of Uberlândia (UFU)**.
- **Eric Anacleto Ribeiro** for the extensive work on CICIDS2017 preprocessing and cleaning.
- The original CICIDS2017 dataset creators (Canadian Institute for Cybersecurity).
