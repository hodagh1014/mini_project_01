# Credit Card Fraud Detection

## 1. Problem Description

### Business Scenario

Credit card fraud is an important problem for banks and financial institutions. A bank needs a system that can automatically detect transactions that may be fraudulent.

In this project, I built a machine learning pipeline to predict whether a transaction is legitimate or fraudulent.

The target variable is:

* 0 → Legitimate
* 1 → Fraud

### Dataset

I used the Credit Card Fraud Detection dataset from Kaggle.

* **Samples:** 284,807
* **Features:** 30
* **Target:** Class
* **Problem:** Binary Classification
* **Fraud ratio:** about 0.17%

The features are:

* Time
* V1 to V28
* Amount

The V1 to V28 features are anonymized numerical features.

---

## 2. Data Analysis

The dataset originally contained 284,807 rows and 31 columns (30 features + the target).

### Dataset Statistics

| Property                          |   Value |
| Original samples                  | 284,807 |
| Features                          |      30 |
| Columns                           |      31 |
| Duplicate records                 |   1,081 |
| Samples after removing duplicates | 283,726 |
| Missing values                    |       0 |

I found 1,081 duplicate records and removed them before training the models.

### Class Distribution

The dataset is highly imbalanced:

| Class          | Percentage |
| Legitimate (0) |   99.8273% |
| Fraud (1)      |    0.1727% |

This means that fraudulent transactions are very rare compared to legitimate transactions.

### Amount

* Mean: 88.35
* Median: 22.00
* Minimum: 0.00
* Maximum: 25,691.16

### Time

* Mean: 94,813.86
* Median: 84,692.00
* Minimum: 0.00
* Maximum: 172,792.00

### Missing Values

No missing values were found in the dataset.

---

## 3. Initial Hypothesis

Before training the models, I had the following expectations.

### Logistic Regression

I expected Logistic Regression to be a good baseline because this is a binary classification problem.

### KNN

I expected KNN to be affected strongly by feature scaling because it uses distance calculations.

### Decision Tree

I expected Decision Tree to be able to find nonlinear relationships, but I also expected it to have a risk of overfitting.

### Neural Network

I expected the Neural Network to be able to learn more complex patterns and possibly achieve better results.

### Important Metric

I expected Recall to be very important because missing a fraudulent transaction can be costly.

However, using Recall alone is not enough because increasing Recall can also increase the number of false alarms. Therefore, Precision and F1-score are also important.

### Predicting Everything as Legitimate

If a model predicts every transaction as legitimate, the Accuracy would still be around 99.83%.

However, it would detect no fraudulent transactions, so its fraud Recall would be 0%.

This shows why Accuracy alone is not a good metric for this problem.

---

## 4. Model Comparison

I trained four models:

1. Logistic Regression
2. KNN
3. Decision Tree
4. Neural Network (MLP)

### Test Results

| Model               |   Accuracy |  Precision | Recall |   F1-Score |
| ------------------- | ---------: | ---------: | -----: | ---------: |
| Logistic Regression |     0.9753 |     0.0564 | 0.8737 |     0.1059 |
| KNN                 |     0.9994 |     0.9559 | 0.6842 |     0.7975 |
| Decision Tree       |     0.9644 |     0.0388 | 0.8526 |     0.0743 |
| **Neural Network**  | **0.9995** | **0.9589** | 0.7368 | **0.8333** |

The Neural Network achieved the highest F1-score on the test set.

It also had the highest Precision among the tested models.

Logistic Regression had the highest Recall, but its Precision was very low. This means that it detected many fraud cases but also produced many false alarms.

### Confusion Matrices

#### Logistic Regression


[[55262  1389]
 [   12    83]]


#### KNN


[[56648     3]
 [   30    65]]


#### Decision Tree


[[54647  2004]
 [   14    81]]


#### Neural Network


[[56648     3]
 [   25    70]]


For the Neural Network:

* True Negatives = 56,648
* False Positives = 3
* False Negatives = 25
* True Positives = 70

So the model detected 70 fraudulent transactions and missed 25. It also produced only 3 false positives.

---

## 5. Cross Validation

I used 5-Fold Stratified Cross Validation because the dataset is highly imbalanced.

| Model               | Mean Precision | Mean Recall |    Mean F1 |
| ------------------- | -------------: | ----------: | ---------: |
| Logistic Regression |         0.0605 |      0.9179 |     0.1135 |
| KNN                 |         0.9157 |      0.7646 | **0.8318** |
| Decision Tree       |         0.0590 |      0.8519 |     0.1103 |
| Neural Network      |         0.8773 |      0.7829 |     0.8245 |

KNN had the highest mean F1-score in cross-validation.

However, on the final test set, the Neural Network had the highest F1-score.

This shows that the best result can be slightly different between cross-validation and the final test set.

---

## 6. Scaling Experiment

I compared KNN with and without feature scaling.

| Model | Scaling         | Precision | Recall |     F1 |
| ----- | --------------- | --------: | -----: | -----: |
| KNN   | Without Scaling |    1.0000 | 0.0211 | 0.0412 |
| KNN   | With Scaling    |    0.9559 | 0.6842 | 0.7975 |

The difference is very large.

Without scaling, KNN had a Recall of only 0.0211.

After scaling, Recall increased to 0.6842 and F1 increased to 0.7975.

This happens because KNN uses distance calculations. Features with larger numerical ranges can have a much larger effect on the distance.

Decision Trees are less affected by scaling because they use conditions such as:

Amount > 100


instead of calculating distances between samples.

---

## 7. Hyperparameter Experiment

For KNN, I tested three different values of k.

| K  |  Precision | Recall |         F1 |
| -- | ---------: | -----: | ---------: |
| 1  |     0.8293 | 0.7158 |     0.7684 |
| 5  | **0.9559** | 0.6842 | **0.7975** |
| 20 |     0.9155 | 0.6842 |     0.7831 |

k=5 produced the highest F1-score.

For k=1 , the model depends heavily on individual nearest samples, so it can have higher variance.

For k=20, the decision becomes smoother because more neighbors are used, but the F1-score was slightly lower.

Based on these experiments, k=5 gave the best result among the tested values.

---

## 8. Threshold Experiment

I tested three classification thresholds:

* 0.3
* 0.5
* 0.7

The threshold experiment was performed using KNN.

| Threshold | Precision |     Recall |         F1 |
| 0.3       |    0.9114 | **0.7579** | **0.8276** |
| 0.5       |    0.9559 |     0.6842 |     0.7975 |
| 0.7       |    0.9836 |     0.6316 |     0.7692 |

When the threshold decreases, the model becomes more likely to classify a transaction as fraud.

This increases Recall because more fraudulent transactions are detected, but it can also decrease Precision because more legitimate transactions may be flagged.

In this experiment, threshold 0.3 gave the highest F1-score and Recall.

---

## 9. Final Model Selection

Based on the test-set results, I selected the **Neural Network (MLP)** as the final model.

Its test results were:

| Metric    | Result |
| Accuracy  | 0.9995 |
| Precision | 0.9589 |
| Recall    | 0.7368 |
| F1-score  | 0.8333 |

The main reason for selecting the Neural Network was its F1-score of 0.8333, which was the highest F1-score among the models on the test set.

The implemented training pipeline selected a threshold of **0.3**.

The threshold experiment itself was performed with KNN. In that experiment, threshold 0.3 gave the best F1-score.

A limitation of this experiment is that the threshold was not separately tested for the Neural Network. Testing different thresholds specifically for the final Neural Network would be a useful improvement.

---

## 10. Data Leakage Prevention

To avoid data leakage, the data was first divided into training and test sets.

The scaler was fitted using the training data and then used to transform both the training and test data.

The test data was not used to fit the scaler.

I also used a stratified train/test split so that the distribution of legitimate and fraudulent transactions was preserved.

The training set contained:

226,980 samples


and the test set contained:


56,746 samples


---

## 11. Model Saving

After training, the final Neural Network model was saved as:

models/nn_model.pkl


The prediction script loads the saved model and uses the same preprocessing process used during training.

---

## 12. Prediction

I also tested the prediction part of the project after training.

The prediction pipeline successfully loaded the saved model and returned a JSON result.

Example output:

{
  "prediction": "Legitimate",
  "class_id": 0,
  "probability": 0.0002445303848578187,
  "threshold": 0.3,
  "status": "success"
}


The fraud probability for this transaction was approximately:


0.02445%


Since this value is below the selected threshold of 0.3, the transaction was classified as legitimate.

---

## 13. Running Instructions

### 1. Clone the repository


git clone https://github.com/hodagh1014/mini_project_01.git
cd mini_project_01


### 2. Create a virtual environment


python -m venv venv


### 3. Activate the virtual environment

On Windows:


venv\Scripts\activate


### 4. Install the requirements


pip install -r requirements.txt


### 5. Train the models


python src/train.py


The training script loads and analyzes the data, removes duplicates,
splits the data, trains the models, evaluates them, performs the
experiments, selects the final model and saves it.

### 6. Run prediction


python src/predict.py


The prediction script loads the saved model and predicts whether a new
transaction is legitimate or fraudulent.

---

## 14. Reflection

### Question 1: Why is Accuracy misleading?

The dataset is highly imbalanced. Only about 0.17% of the transactions are fraudulent.

Because of this, a model can predict almost everything as legitimate and still get very high Accuracy.

For example, predicting all transactions as legitimate gives an Accuracy of about 99.83%, but the model detects zero fraud cases.

Therefore, Precision, Recall, F1-score and the Confusion Matrix are more useful for this problem.

### Question 2: What is the trade-off between detecting more fraud and generating more false alarms?

When the threshold is decreased, the model usually detects more fraudulent transactions.

This increases Recall, but it can also increase false positives and reduce Precision.

When the threshold is increased, the model becomes more conservative. This can improve Precision but may cause more fraudulent transactions to be missed.

For fraud detection, missing a fraud transaction can be more costly than investigating a false alarm, so Recall is especially important.

However, Precision should also be considered because too many false alarms can make the system difficult to use.

### Question 3: What would I improve with one additional week?

If I had one additional week, I would improve the project in several ways:

1. Test methods for handling class imbalance such as SMOTE.
2. Try more advanced models such as Random Forest or XGBoost.
3. Perform more hyperparameter tuning.
4. Create additional useful features from the transaction data.
5. Test different thresholds specifically for the final Neural Network.
6. Add model explainability using SHAP.
7. Improve the prediction system and possibly create a FastAPI API.
8. Add model monitoring for future changes in the data.

---

## 15. Final Summary

The main result of this project was that Accuracy alone was not enough to evaluate a fraud detection model.

KNN showed how important feature scaling can be, while Logistic Regression showed that high Recall can come with many false positives.

The Neural Network achieved the best F1-score on the test set:


F1 = 0.8333


with:


Precision = 0.9589
Recall    = 0.7368


The project also included 5-fold stratified cross-validation, scaling analysis, KNN hyperparameter tuning, threshold analysis and a working prediction pipeline.

Overall, this project helped me understand the complete machine learning workflow from data preparation to model training, evaluation and prediction.
