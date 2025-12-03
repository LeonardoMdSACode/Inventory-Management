## About Inventory-AllRegression.ipynb

# Comprehensive Regression Analysis for Profit Prediction

## 🚀 Project Overview

This project aims to predict the **Profit** generated from sales transactions in the `ML-Dataset.csv` dataset. Since Profit is a continuous numerical variable, this task is framed as a **regression problem**.

The primary goal is to perform a rigorous comparison of a large and diverse set of popular machine learning regression models available in the Python ecosystem (`scikit-learn`, `XGBoost`, `LightGBM`) to identify the best-performing algorithm based on key evaluation metrics ($\text{R}^2$ and MAE).

The analysis pipeline includes:

1. **Data Cleaning & Leakage Mitigation:** Removing direct profit components and irrelevant identifiers.

2. **Feature Engineering:** Extracting temporal features from the `OrderDate`.

3. **Comprehensive Modeling:** Training 16 distinct regression algorithms.

4. **Evaluation:** Calculating and visualizing performance metrics.

## 💾 Data Preparation and Feature Selection

The original dataset contained 200,000 records and 26 features.

### 1. Target Variable

* **Target:** `Profit` (Continuous, numerical).

### 2. Data Leakage Mitigation (Crucial Step)

To ensure the model learns generalized patterns rather than simply reproducing a known formula, features that mathematically define profit were deliberately **removed** from the training set.

| Removed Leakage Features | Reason |
| :--- | :--- |
| `ProductStandardCost` | Direct component of the profit calculation. |
| `ProductListPrice` | Direct component of the profit calculation. |

### 3. Feature Engineering

The `OrderDate` column was converted to a datetime object, and three new numerical features were extracted to capture seasonality and time-based patterns:

* `Order_Month`
* `Order_Year`
* `Order_Weekday`

### 4. Preprocessing Pipeline

All remaining features were processed using a `ColumnTransformer` within a scikit-learn pipeline:

* **Numerical Features** (`CustomerCreditLimit`, `OrderItemQuantity`, etc.) were scaled using `StandardScaler`.
* **Categorical Features** (`RegionName`, `CategoryName`, `Status`, etc.) were converted using `OneHotEncoder`.

### 5. Training and Testing Split

The prepared dataset was split into training (80%) and testing (20%) sets to ensure models are evaluated on unseen data.

## 🧠 Regression Models Tested

The following 16 regression models were tested for performance robustness on the dataset:

| Model Category | Model Name | Description |
| :--- | :--- | :--- |
| **Linear** | Linear Regression | The simplest baseline model. |
| **Regularized** | Ridge Regression (L2) | Linear model with L2 regularization (prevents large coefficients). |
| **Regularized** | Lasso Regression (L1) | Linear model with L1 regularization (enforces sparsity/feature selection). |
| **Regularized** | ElasticNet Regression | Combination of L1 and L2 regularization. |
| **Bayesian/Robust** | Bayesian Ridge Regression | Linear model incorporating probabilistic approach. |
| **Bayesian/Robust** | TheilSen Regressor | Robust non-parametric method, less sensitive to outliers. |
| **Bayesian/Robust** | Huber Regressor | Linear model that reduces the weight of outliers. |
| **Tree-based** | Decision Tree Regressor | Non-linear model that splits data based on feature values. |
| **Ensemble (Bagging)** | Random Forest Regressor | Multiple decision trees averaging predictions. |
| **Ensemble (Boosting)** | Gradient Boosting Regressor | Sequentially builds weak models to correct errors. |
| **Ensemble (Fast Boost)** | **Hist Gradient Boosting** | Faster, optimized Gradient Boosting implementation. |
| **Ensemble (Extreme)** | **XGBoost Regressor** | Highly optimized and widely used gradient boosting framework. |
| **Ensemble (Extreme)** | **LightGBM Regressor** | Highly efficient, histogram-based gradient boosting framework. |
| **Neighbors/Kernel** | K-Nearest Neighbors | Instance-based learning (predictions based on nearest neighbors). |
| **Neighbors/Kernel** | Support Vector Regressor (SVR) | Non-linear model using kernel tricks to map data to a higher dimension. |
| **Ensemble (Boosting)** | AdaBoost Regressor | Sequentially builds models to correct errors of previous models. |

## 📊 Evaluation Results

The models were evaluated using three primary regression metrics:

| Metric | Goal | Meaning |
| :--- | :--- | :--- |
| $\text{R}^2$ Score | **Maximize** (Closer to 1.0) | Proportion of the variance in the dependent variable that is predictable from the independent variables. |
| MAE (Mean Absolute Error) | **Minimize** (Closer to 0) | The average absolute difference between the predicted profit and the actual profit (measured in \$). |
| MSE (Mean Squared Error) | **Minimize** (Closer to 0) | The average squared difference between predicted and actual profit. Penalizes larger errors more heavily. |

### Model Performance Table (Sorted by $\text{R}^2$ Score)

| Model | R2 | MAE | MSE |
| :--- | :--- | :--- | :--- |
| **Decision Tree Regressor** | **0.637** | **70.46** | **17161.62** |
| Random Forest Regressor | 0.510 | 74.47 | 23195.79 |
| XGBoost Regressor | 0.461 | 78.14 | 25508.33 |
| Gradient Boosting Regressor | 0.434 | 78.01 | 26762.09 |
| Linear Regression | 0.357 | 99.06 | 30395.77 |
| TheilSen Regressor | 0.356 | 99.08 | 30468.30 |
| Lasso Regression (L1) | 0.315 | 101.26 | 32421.40 |
| LightGBM Regressor | 0.295 | 102.78 | 33324.59 |
| Ridge Regression (L2) | 0.287 | 104.49 | 33729.19 |
| Hist Gradient Boosting | 0.247 | 107.77 | 35597.38 |
| Huber Regressor | 0.165 | 112.19 | 39509.75 |
| ElasticNet Regression | 0.109 | 128.18 | 42159.89 |
| AdaBoost Regressor | -0.013 | 168.54 | 47900.91 |
| Bayesian Ridge Regression | -0.018 | 138.19 | 48158.93 |
| K-Nearest Neighbors | -0.121 | 148.43 | 53037.02 |
| Support Vector Regressor (SVR) | -0.158 | 139.95 | 54748.65 |

### 🏆 Conclusion and Best Model

The **Decision Tree Regressor** is the best-performing model based on the current run, followed closely by Random Forest and XGBoost.

* **Best R2 Score:** $\mathbf{0.637}$ (Decision Tree Regressor). This indicates that the model explains about 63.7% of the variance in the Profit variable.

* **Best MAE:** $\mathbf{\$70.46}$ (Decision Tree Regressor). On average, the predicted profit is off by approximately $\$70.46$ per transaction.

This result suggests that while tree-based models are effective at capturing the non-linear structure of the profit data (outperforming all linear and kernel methods), there is still significant unexplained variance ($\sim 36.3\%$). Further efforts should focus on:

1. **Feature Engineering:** Creating more sophisticated features that capture complex interactions not visible to the current model.

2. **Hyperparameter Tuning:** Optimizing the hyperparameters (especially for XGBoost and Random Forest) to potentially improve their performance and close the gap with the Decision Tree.

3. **Outlier Analysis:** Investigating the large residuals (implied by the high MSE/MAE) to see if outliers are skewing the results.

---

## About Inventory-TensorFlow.ipynb


# Deep Learning Regression for Business Profit Prediction

This document summarizes the steps taken in the provided Python script, which focuses on predicting business profit using various Deep Neural Network (DNN) architectures. The workflow covers data preparation, feature engineering, advanced preprocessing, model definition, training, and comparative evaluation.

---

## 1. Project Setup and Data Preparation

- **Libraries:** pandas, tensorflow/keras, sklearn, seaborn.
- **Reproducibility:** Seeds are set for NumPy and TensorFlow to ensure consistent results.
- **Data Source:** `ML-Dataset.csv` (400 rows, 28 columns)
- **Target Variable:** `Profit` (a continuous numerical value)

---

## 2. Feature Selection and Engineering

### Data Cleaning and Leakage Mitigation
- Non-predictive columns and those causing potential data leakage (`ProductStandardCost`, `ProductListPrice`, customer/employee PII) were dropped.

### Temporal Feature Engineering
- `OrderDate` column was converted to datetime.
- Three new numerical features were extracted, then the original column was dropped:
  - `Order_Month`
  - `Order_Year`
  - `Order_Weekday` (Day of the week)

---

## 3. Preprocessing Pipeline

- Data split: 80% training, 20% testing.
- **ColumnTransformer** used to prepare features:
  - **Numerical Features:** Scaled using `StandardScaler` (e.g., `CustomerCreditLimit`)
  - **Categorical Features:** Encoded using `OneHotEncoder`
- After preprocessing, the input dimension for the neural networks is **498 features**.

---

## 4. Deep Learning Architectures

Four distinct DNN models were defined and trained, all incorporating **L2 Regularization (0.001)** and using a **Huber Loss function**:

| Model Name                   | Type                  | Layers               | Key Features                                                |
|-------------------------------|---------------------|--------------------|------------------------------------------------------------|
| `Simple_DNN_3_Layers_L2`     | Shallow              | 3 Dense Layers      | Baseline architecture (64, 32 units)                       |
| `Deep_DNN_6_Layers_L2`       | Deep                 | 6 Dense Layers      | High capacity (256, 128, 64, 64, 32 units)                |
| `Regularized_DNN_Robust`     | Deep, Robust         | 3 Dense Layers + Regulators | Incorporates BatchNormalization and Dropout (0.3)       |
| `Wide_Network_L2`             | Wide                 | 2 Dense Layers      | Fewer layers but high width (512 units)                    |

---

## 5. Training and Results

- All models trained for up to **300 epochs** using `EarlyStopping` and `ReduceLROnPlateau` callbacks for stabilization.

### Model Performance Comparison (Test Set)

| Model                        | MAE (Mean Absolute Error) | MSE (Mean Squared Error) | R² (Coefficient of Determination) |
|-------------------------------|---------------------------|-------------------------|----------------------------------|
| `Wide_Network_L2`             | 105.86                    | 543474                  | 0.081                            |
| `Simple_DNN_3_Layers_L2`      | 179.46                    | 269499                  | -0.469                           |
| `Deep_DNN_6_Layers_L2`        | 179.83                    | 569775                  | -0.475                           |
| `Regularized_DNN_Robust`      | 206.09                    | 587848                  | -0.857                           |

---

## Conclusion

- **Best Model:** `Wide_Network_L2` achieved the lowest MAE (**105.87**) and highest $R^2$ (**0.081**).  
- **Observation:** The generally low and often negative $R^2$ scores indicate that the current features are insufficient for highly accurate profit prediction. Models perform worse than predicting the average profit.  
- **Key Recommendation:** Focus on advanced feature engineering or testing highly effective non-linear models like Gradient Boosting Machines (XGBoost, LightGBM) before iterating further on DNN architectures.
- 
PS: Keep in mind the values can vary depending on the run.

---

## About Inventory-PyTorch.ipynb

# PyTorch Deep Learning Regression Model Comparison for Profit Prediction

## 1. Project Overview

This project implements and compares four different Deep Neural Network (DNN) architectures built using PyTorch for a regression task. The goal is to predict the **Profit** of a business based on various transactional and customer features from the provided dataset (*ML-Dataset.csv*).

The script follows a full machine learning pipeline:  
data cleaning, feature engineering, standardized preprocessing using `sklearn.ColumnTransformer`, conversion to PyTorch Tensors and DataLoaders, training with custom callbacks (Early Stopping, Learning Rate Scheduling), and final performance evaluation.

---

## 2. Data Preprocessing Pipeline

To prepare the structured data for neural networks, a comprehensive preprocessing strategy was used:

| Stage                     | Action                                                                 | Libraries / Tools |
|--------------------------|-------------------------------------------------------------------------|-------------------|
| Feature Engineering      | Extracted `Order_Month`, `Order_Year`, `Order_Weekday` from `OrderDate`. | Pandas            |
| Data Leakage Mitigation  | Dropped columns directly related to the target (e.g., `ProductStandardCost`, `ProductListPrice`) and non-predictive identifiers. | Pandas |
| Scaling                  | Numerical features scaled using `StandardScaler`.                       | Scikit-learn      |
| Encoding                 | Categorical features transformed using `OneHotEncoder`.                 | Scikit-learn      |
| Data Split               | Training, Validation (20% of train), Test (20% of total).                | Scikit-learn      |
| PyTorch Conversion       | NumPy arrays converted to `torch.tensor` and wrapped in `DataLoader`.    | PyTorch           |

---

## 3. PyTorch Model Architectures

Four distinct neural networks were developed to explore depth, width, and regularization.

| Model Name                   | Description                                   | Key Features |
|-----------------------------|-----------------------------------------------|--------------|
| **Simple_DNN_3_Layers_L2**  | Baseline, minimal depth.                     | 3 Dense layers (64, 32, 1). Uses **L2 Regularization** (`λ = 0.001`). |
| **Deep_DNN_6_Layers_L2**    | Deep, high complexity.                       | 6 Dense layers up to 256 neurons. Uses **L2 Regularization**. |
| **Regularized_DNN_Robust**  | Deep + heavy regularization.                 | BatchNorm + Dropout (0.3). Uses **RMSprop**. |
| **Wide_Network_L2**         | Shallow but wide.                            | 2 Dense layers (512, 1). Uses Dropout (0.2) + **L2 Regularization**. |

---

## 4. Training Configuration

| Parameter        | Value             | Details |
|------------------|-------------------|---------|
| **Loss Function** | `nn.SmoothL1Loss` | Huber-like, robust to outliers. |
| **Optimizer**     | Adam (default) or RMSprop | LR = 0.001, Weight Decay = 0.001. |
| **Epochs**        | 300 (max)        | Early stopping applied. |
| **Batch Size**    | 64               | Efficient GPU/CPU use. |
| **Early Stopping** | Patience = 20    | Stops if no val-loss improvement. |
| **LR Scheduler**  | `ReduceLROnPlateau` | Reduces LR by factor 0.5 if no improvement for 5 steps. |

---

## 5. Model Comparison Results

| Model                      | MAE              | MSE               | R² Score |
|---------------------------|------------------|-------------------|----------|
| **Deep_DNN_6_Layers_L2**  | 101.749          | 41127.092         | 0.130    |
| **Wide_Network_L2**       | 105.075          | 41634.660         | 0.120    |
| **Regularized_DNN_Robust**| 106.968          | 43557.861         | 0.079    |
| **Simple_DNN_3_Layers_L2**| 128.954          | 53063.318         | -0.122   |

---

## Summary

- The **Deep_DNN_6_Layers_L2** achieved the best performance with an R² score of **0.130** and MAE of **101.75**.  
- The MAE indicates the model’s predictions are off by about **\$101.75** on average.  
- The overall low R² values suggest the relationship between features and target is complex, nonlinear, or strongly influenced by unobserved variables.  
- PyTorch allowed granular control over training loops, losses, and callbacks, enabling explicit implementations of regularization and optimization strategies.

