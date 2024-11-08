
# TCR/MHC Prediction Models

This repository contains scripts designed to train, evaluate, and visualize models for TCR/MHC interaction predictions. These scripts utilize machine learning classifiers and transform amino acid frequency data to predict TCR/MHC interactions. 

## Files

- **`beta.py`**: This script trains and evaluates three classifiers (Linear SVC, Logistic Regression, and Random Forest) on a dataset of TCR/MHC interactions. It loads the training and testing datasets, performs data transformation, and plots Receiver Operating Characteristic (ROC) curves to evaluate model performance.

- **`alpha-beta.py`**: Similar to `beta.py`, this script includes expanded features in the dataset, working on a larger feature set for training and testing the TCR/MHC interaction prediction models.

## Requirements

The following Python packages are required to run the scripts:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Dataset Format

To use the scripts, the training and testing datasets must be in a specific format, as detailed below. You can use **[ANARCI](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabpred/anarci/)** (Antibody Numbering and Receptor ClassIfication) to determine the positions in the TCR sequence that correspond to the required columns.

- **Columns**:
  - Initial columns contain identifiers and gene names:
    - `complex.id` or `MHC Allele Names`: Identifiers for each sample.
    - `TRA_Gene` and `TRB_Gene`: Gene names for the alpha (TRA) and beta (TRB) chains.
  - **TRA and TRB columns**: Each subsequent column represents an amino acid at a specific position in the TRA or TRB chain, with columns named as follows:
    - `TRA_27`, `TRA_28`, `TRA_29`, ..., `TRA_65` for the alpha chain.
    - `TRB_27`, `TRB_28`, `TRB_29`, ..., `TRB_85` for the beta chain.
  - **Type**: The final column, `Type`, indicates the binary label for interaction status (0 or 1).

### Example

Here is a sample of the dataset format:

| complex.id | TRA_Gene | TRB_Gene | TRA_27 | TRA_28 | ... | TRB_84 | TRB_85 | Type |
|------------|----------|----------|--------|--------|-----|--------|--------|------|
| 1          | TRAV26-1*01 | TRBV13*01 | T      | I      | ... | S      | D      | 0    |
| 2          | TRAV20*01   | TRBV13*01 | V      | S      | ... | S      | D      | 0    |

Using **ANARCI**, you can obtain the specific amino acid positions for your TCR sequences and map them to the correct columns (`TRA_27`, `TRA_28`, ..., `TRB_85`) in the dataset to ensure compatibility with the scripts.

## Usage

Both scripts accept command-line arguments for input paths and parameters. The general syntax is:

`
python script_name.py <train_path> <test_path> [--output_img <filename>] 
`

- **`train_path`**: Path to the training dataset in CSV format.
- **`test_path`**: Path to the testing dataset in CSV format.
- **`--output_img`** (optional): Filename to save the output ROC curve plot. Default is `roc_curve.png`.

### Example

`
python beta.py data/Training_alpha_and_beta_final.csv data/Testing_alpha_and_beta.csv --output_img beta_roc.png
`

`
python alpha-beta.py data/Training_alpha_and_beta_final.csv data/Testing_alpha_and_beta.csv --output_img alpha_beta_roc.png
`

## How It Works

1. **Data Loading**: The scripts load both training and testing datasets, selecting specific columns for model input.
2. **Data Transformation**: A transformation function converts amino acid frequency data for class 0 and class 1, calculates frequency ratios, and applies a logarithmic transformation.
3. **Model Training and Evaluation**: Three classifiers (Linear SVC, Logistic Regression, Random Forest) are trained and cross-validated. ROC curves are plotted to compare the performance of each classifier.

## Output

Each script generates an ROC curve plot saved as an image file. The plot includes the following details:
- Mean AUC (Area Under the Curve) across cross-validation folds.
- Test AUC for evaluating the final model on the test dataset.

---

