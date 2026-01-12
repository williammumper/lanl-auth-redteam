# LANL Cyber Anomaly Detection

Machine learning pipeline for detecting authentication anomalies in the Los Alamos National Labs cyber dataset.

## Overview
The LANL dataset contains 3,000,000+ daily authentication events from enterprise network activity. This project builds an ensemble + LSTM pipeline to identify anomalous patterns that could indicate insider threats or compromised accounts.

## Problem
- Massive volume of authentication logs (3M+ events/day)
- Need to identify rare anomalous behavior among normal activity
- False positive rate can balloon very quickly with the number of events

## Approach
**Ensemble Methods:**
- Isolation Forest
- One-Class SVM  
- Local Outlier Factor

**Sequential Modeling:**
- LSTM for user auth log pattern analysis

**Visualization:**
- Web interface for real-time anomaly metrics

## Results
- Processes dataset in manageable chunks - resource friendly
- Flags statistical anomalies but has limited effectiveness on sophisticated attacks
- Initial apprach highlighted the gap between staitstical methods and real threat detection
- Identified areas for improvement: feature engineering, user catalogs

## Tech Stack
Pandas, Numpy, Scikit-learn, PyTorch, Matplotlib, Seaborn, Jupyter

## Key Learnings
- Real-world importance of solid feature engineering
- Tinkering with ensemble methods and LSTM on a massive dataset
- Rapid prototyping and debugging

## Datasets
- `auth.txt.gz` - authentication logs
- `redteam.txt.gz` - labeled red team activity