# CSE676 Group 21: House Price Prediction with Deep Learning for Tabular Data

This repository contains our CSE676 course project on predicting house prices using deep learning models for tabular regression. We evaluate multiple architectures (FNN/DNN, Wide & Deep, TabNet) and compare optimizers (SGD, Adam, AdamW, Ranger, Lion). Our best configuration is **TabNet + Lion**, achieving strong RMSE and R² on the test set, while also providing feature interpretability through attentive masks.

## Motivation (Movie Context)
Our project is inspired by a movie scenario where two sisters inherit property in Mallorca but are deceived into selling below market value. This highlights how traditional valuation can be vulnerable under information asymmetry. We aim to improve transparency in pricing by learning data-driven estimates of fair value using deep learning.

## Dataset
- **Dataset:** California Housing
- **Target:** `median_house_value`
- Includes geographic + demographic + housing-structure features
- Contains a coastal category feature (e.g., `ocean_proximity`) enabling analysis of coastal influence.

## Feature Engineering
We create meaningful ratio features:
- `rooms_per_household = total_rooms / households`
- `bedrooms_per_room = total_bedrooms / total_rooms`
- `population_per_household = population / households`

## Models Implemented
- **FNN / DNN:** Fully-connected regression baselines with ReLU activations
- **Wide & Deep:** Linear “wide” part + deep neural network part
- **TabNet:** Attentive interpretable model for tabular learning (decision steps + sparse feature masks)

## Loss Function
We treat this as regression and train with standard regression loss (e.g., **MSE**).  
Note: **ReLU is an activation function, not a loss function.**

## Optimization Algorithms
We compare:
- SGD
- Adam
- AdamW
- Ranger
- Lion (best-performing optimizer in our experiments)

## Evaluation Metrics
We report:
- MAE
- RMSE
- R²

## Results (Summary)
Best per architecture (test set):
- FNN: Lion
- DNN: AdamW
- Wide & Deep: Lion
- **TabNet: Lion (best overall)**

