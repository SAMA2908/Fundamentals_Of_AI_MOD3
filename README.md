# ATP Match Analysis with Bayesian Networks

## Project Overview
This project utilizes **Bayesian Networks** to analyze and predict the outcomes of ATP tennis matches from **2014 to 2024**. The primary objective is to understand the causal dynamics driving match results, with a specific focus on predicting "Upsets" (when the favorite loses).

The model evaluates the impact of three macro-factors on the final result:
1.  **Status (Ex-Ante):** Pre-match knowledge such as Ranking Difference, Age, and Height.
2.  **Performance (Ex-Post):** Actual in-match statistics including Aces, Double Faults, and Service Points Won.
3.  **Context:** Environmental factors, specifically the playing **Surface**.

## Project Structure
The analysis compares four different probabilistic models built using the `pgmpy` library:
* **Expert Model:** A causal graph manually defined based on tennis domain knowledge.
* **Naive Bayes:** A statistical baseline assuming feature independence.
* **Variant Model:** An expert model variant excluding physical attributes to test redundancy.
* **Learned Model:** A structure learned automatically from the data using Hill Climbing Search.

---

## Execution Order
To replicate the analysis, **you must execute the notebooks in the following logical order**, as the output of one file serves as the input for the next.

### 1. Data Preparation (`data_preparation.ipynb`)
**Mandatory First Step.**
* **Input:** Raw yearly datasets (`atp_matches_2014.csv` to `2024.csv`).
* **Process:**
    * Cleans the data and handles missing values.
    * **Feature Engineering:** Computes differentials (e.g., Rank Diff, Ace Diff) and advanced features like `Fav_On_Worst_Surface` and `Fav_Recent_Form`.
    * **Discretization:** Transforms continuous variables into categorical bins (e.g., Low/Medium/High) required for Discrete Bayesian Networks.
    * **Splitting:** Performs a strict time-series split:
        * *Training:* 2014-2023
        * *Testing:* 2024
* **Output:** Generates the cleaned files in the `processed_data/` directory (`atp_matches_training.csv`, `atp_matches_test_2024.csv`).

### 2. Exploratory Data Analysis (`exploratory_analysis.ipynb`)
**Optional but Recommended.**
* **Input:** The processed data files generated in Step 1.
* **Process:**
    * Visualizes feature distributions (Win/Loss correlations).
    * Analyzes physical stats and ranking impact.
    * **Data Drift Check:** Crucial step to verify statistical consistency between the Training years and the Test year (2024).

### 3. Bayesian Modeling (`bayesian_models.ipynb`)
**Final Analysis Step.**
* **Input:** The processed data files generated in Step 1.
* **Process:**
    * Constructs the four Bayesian Network topologies.
    * Learns parameters (CPDs) using Maximum Likelihood Estimation.
    * **Validation:** Evaluates accuracy against the 2024 season ground truth.
    * **Inference & Scenarios:** Performs "What-If" analysis (e.g., "How does the probability of winning change if a Strong Favorite plays on their worst surface?").

---

## Installation & Requirements

1.  Clone the repository.
2.  Ensure you have Python 3.10+ installed.
3.  Install the dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt