# Property-Insurance-Risk-Engine
Claim Probability Scoring with XGBoost End-to-End Underwriting Model targeting Risk Exposure and Imbalance Handling.

## Project Overview

As the Lead Data Analyst / Data Scientist on this project, I developed a risk-scoring model to estimate the probability that a building will experience at least one insurance claim within a given policy period.

Unlike traditional binary classification (Claim vs No Claim), this project focuses on probabilistic risk modeling, enabling insurance companies to:

Set fair, risk-adjusted premiums

Improve capital reserve planning

Proactively identify high-risk properties

The final model was built using XGBoost and achieved a ROC-AUC score of ~0.71, demonstrating strong performance even on a highly imbalanced dataset.

## Problem Statement

Insurance underwriting is fundamentally a probability problem.

A “safe” building is not defined only by structure, but by:

Exposure duration (time insured)

Physical characteristics

Maintenance and security indicators

## Key Challenges

The dataset was highly imbalanced (~22% of buildings had claims).

A naïve model could predict “No Claim” for all observations and still achieve high accuracy — with zero business value.

Objective

Maximize Recall (capture as many real claims as possible)

Control false positives

Optimize for ROC-AUC, not accuracy

## Data Description

The dataset contains detailed attributes of insured buildings:

## Feature	Description
Pol_Duration	Insured period (e.g., 1.0 = full year, 0.5 = 6 months)

Dimensions	Building size (square meters)

Age	Derived from date of occupancy

Location	Anonymized geographic code

Garden, Fence, Painted	Indicators of maintenance and security

Claim	Target variable (1 = claim occurred, 0 = no claim)


## Feature Engineering Strategy

Raw features alone were insufficient. Domain knowledge was applied to engineer risk-aware features.

### Exposure Index (High Impact Feature)

Risk increases with both asset size and exposure duration

Formula:

Exposure Index = Building Dimension × Insured Period


Impact:

Became the most important predictor in the final model

Validated the intuition that total asset exposure drives claim probability

### Security Score

Individual safety features have limited predictive power alone. Combined, they reflect maintenance quality and ownership responsibility.

Logic:

Security Score = Is_Painted + Is_Fenced + Has_Garden


## Insight:

Buildings with low security scores showed significantly higher claim risk

### Lifecycle Binning (Outlier Handling)

An extreme outlier was present (Building Age = 469 years).

Instead of deleting it, I applied domain-driven binning:

Category	Age Range
New	          0–5
Modern	      6–30
Mid-Life	    31–60
Aging	        61–100
Heritage	    100+

This preserved information while improving model stability.

## Modeling & Methodology

Algorithm Choice: XGBoost Classifier

Selected for its ability to:

Handle missing values natively

Optimize directly for probability ranking (ROC-AUC)

Manage class imbalance via scale_pos_weight

### The “Geo_Code” Incident (Key Learning)

Initial Attempt:

Applied target encoding to Geo_Code

### Problem:

High training AUC, but validation performance collapsed

Model predicted near-zero probability for almost all cases

### Diagnosis:

Severe overfitting and data leakage

Model learned noisy location signals instead of structural risk factors

### Solution:

Removed Geo_Code entirely

Forced the model to rely on safe, generalizable features (exposure, size, age, security)

### Outcome:

Validation AUC jumped from ~0.50 → ~0.71

Model became stable and reliable

## Results & Evaluation

Metric

ROC-AUC: ~0.71

### Interpretation:
The model is effective at ranking buildings from lowest to highest risk, which is ideal for underwriting and pricing decisions.

Confusion Matrix (Threshold = 0.25)

Claims are expensive — missing one is worse than a false alarm.
The decision threshold was lowered from 0.50 to 0.25.

### Predicted No Claim	Predicted Claim
Actual No Claim	4,089	1,437 (False Alarms)
Actual Claim	726	908 (Caught!)

## Business Impact:

Successfully identified 908 high-risk buildings that would likely be missed by a standard classifier.

### How to Run the Project

#### Clone the Repository
git clone https://github.com/yourusername/insurance-claim-prediction.git

#### Install Dependencies
pip install pandas numpy xgboost scikit-learn matplotlib seaborn

#### Run the Notebook

Open Insurance_Model_Final.ipynb and run all cells.

### Future Improvements

Hyperparameter Tuning: GridSearch on learning_rate and max_depth to gain additional AUC

Geo-Clustering: Replace raw location codes with K-Means risk clusters

Ensembling: Stack XGBoost with CatBoost for improved robustness
