# Experiments Report

This report contains the experiments I performed during the Credit Card Fraud Detection project.

The main goal of these experiments was to understand how different preprocessing choices, hyperparameters, classification thresholds, and models affect fraud detection performance.

---

## 1. Scaling Experiment (KNN)

KNN is a distance-based algorithm, so I expected feature scaling to have a significant effect on its performance.

I compared KNN with and without feature scaling.

| Model | Scaling         | Precision | Recall |F1-Score|
| KNN   | Without Scaling | 1.0000    | 0.0211 | 0.0412 |
| KNN   | With Scaling    | 0.9559    | 0.6842 | 0.7975 |

### Analysis

The difference between the two results is very large.

Without scaling, KNN achieved a Precision of 1.0, but its Recall was only 0.0211. This means that the model detected very few fraudulent transactions.

After applying scaling, Recall increased from 0.0211 to 0.6842, and F1-score increased from 0.0412 to 0.7975.

The reason is that KNN calculates distances between samples. If features have very different ranges, features with larger values can have a stronger effect on the distance.

For example, the Amount feature can have values up to 25,691, while many of the anonymized features have much smaller ranges.

### Conclusion

Feature scaling is very important for KNN.

The results clearly show that scaling improved the ability of KNN to detect fraudulent transactions.

---

## 2. Hyperparameter Experiment (KNN)

For the second experiment, I tested different values of k in KNN.

The tested values were:

- k = 1
- k = 5
- k = 20

| K Value | Precision | Recall |F1-Score|
| 1       | 0.8293    | 0.7158 | 0.7684 |
| 5       | 0.9559    | 0.6842 | 0.7975 |
| 20      | 0.9155    | 0.6842 | 0.7831 |

### Analysis

With k=1, the model only looks at the closest sample. This can make the model more sensitive to individual training samples and can lead to higher variance.

With k=20, the model considers more neighbors. This makes the decision smoother, but in this experiment the F1-score was slightly lower.

The highest F1-score was obtained with k=5.

### Conclusion

Among the tested values, k=5 provided the best F1-score and the best balance between Precision and Recall.

Therefore, I selected k=5 for the KNN experiments.

---

## 3. Classification Threshold Experiment

Most classification models use a default threshold of 0.5.

I tested three different thresholds:

- 0.3
- 0.5
- 0.7

The experiment produced the following results:

| Threshold | Precision | Recall | F1-Score |
| 0.3       | 0.9114    | 0.7579 | 0.8276   |
| 0.5       | 0.9559    | 0.6842 | 0.7975   |
| 0.7       | 0.9836    | 0.6316 | 0.7692   |

### Analysis

When the threshold is decreased from 0.5 to 0.3, the model becomes more likely to classify a transaction as fraudulent.

As a result:

- Recall increases.
- More fraudulent transactions are detected.
- Precision decreases.
- More false alarms can be generated.

When the threshold is increased to 0.7:

- Precision increases.
- Recall decreases.
- The model becomes more conservative about predicting fraud.

In this experiment, threshold 0.3 produced the highest F1-score of 0.8276 and the highest Recall of 0.7579.

### Conclusion

For the threshold experiment, 0.3 gave the best result among the three tested thresholds.

However, this threshold experiment was performed using the KNN model. The final selected model was the Neural Network.

Therefore, a useful improvement would be to perform a separate threshold experiment specifically for the final Neural Network.

---

## 4. Cross Validation Results

I used 5-Fold Stratified Cross Validation.

Stratification is important for this dataset because the fraud class is very small compared to the legitimate class.

The results were:

| Model               | Mean Precision | Mean Recall | Mean F1   |
| Logistic Regression | 0.0605         | 0.9179      | 0.1135    |
| KNN                 | 0.9157         | 0.7646      | **0.8318**|
| Decision Tree       | 0.0590         | 0.8519      | 0.1103    |
| Neural Network      | 0.8773         | 0.7829      | 0.8245    |

### Analysis

Logistic Regression had the highest mean Recall (0.9179), which means it detected a large percentage of fraudulent transactions.

However, its Precision was very low (0.0605). This means that it also produced a large number of false alarms.

KNN had the highest mean F1-score of 0.8318.

The Neural Network also performed well, with a mean F1-score of 0.8245.

The cross-validation results were slightly different from the final test-set results. On the test set, the Neural Network achieved the highest F1-score.

### Conclusion

KNN had the best mean F1-score in cross-validation, while Logistic Regression had the highest Recall.

The Neural Network showed a good balance between Precision and Recall and performed best on the final test set.

---

## 5. Final Model Selection

Based on the final test results, I selected the **Neural Network (MLP)** as the final model.

### Performance

| Metric    | Result  |
| Accuracy  | 0.9995  |
| Precision | 0.9589  |
| Recall    | 0.7368  |
| F1-Score  | 0.8333   |

### Confusion Matrix

[[56648     3]
 [   25    70]]

This means:

- True Negatives = 56,648
- False Positives = 3
- False Negatives = 25
- True Positives = 70

The model correctly identified 70 fraudulent transactions and missed 25 fraudulent transactions.

It also produced only 3 false positives.

### Why did I select the Neural Network?

The main reason was its F1-score.

The Neural Network achieved an F1-score of 0.8333, which was the highest test-set F1-score among the four models.

It also achieved a very high Precision of 0.9589.

Although Logistic Regression and Decision Tree had higher Recall, their Precision values were very low. Therefore, they produced many false alarms.

The Neural Network provided a better overall balance between Precision and Recall.

### Selected Threshold

The training pipeline selected a threshold of 0.3.

However, the threshold experiment itself was performed using KNN. Therefore, the threshold selection for the Neural Network should be considered a limitation of the current experiment.

A better approach would be to test different thresholds specifically on the Neural Network before deploying it.

---

## 6. Model Comparison Summary

The final test-set results were:

| Model               | Accuracy | Precision | Recall | F1-Score |
| Logistic Regression | 0.9753   | 0.0564    | 0.8737 | 0.1059   |
| KNN                 | 0.9994   | 0.9559    | 0.6842 | 0.7975   |
| Decision Tree       | 0.9644   | 0.0388    | 0.8526 | 0.0743   |
| **Neural Network**  |**0.9995**| **0.9589**| 0.7368 |**0.8333**|

### Overall Analysis

The results show that Accuracy is not enough to compare the models.

For example, Logistic Regression had an Accuracy of 0.9753, but its F1-score was only 0.1059 because its Precision was very low.

Decision Tree had a similar problem.

KNN and Neural Network had much better Precision and F1-score.

The Neural Network achieved the highest F1-score on the test set, so I selected it as the final model.

---

## 7. Main Findings

From the experiments, I found several important points.

### 1. Scaling is very important for KNN

Without scaling, KNN performed very poorly in terms of Recall and F1-score.

After scaling, its F1-score increased from 0.0412 to 0.7975.

### 2. The value of K affects the result

Among k=1 , k=5 , and k=20 , k=5 produced the highest F1-score.

### 3. Threshold changes the Precision-Recall trade-off

Lowering the threshold increased Recall but reduced Precision.

For this fraud detection problem, increasing Recall can be useful because missing fraudulent transactions can be costly.

### 4. Accuracy can be misleading

Because only about 0.17% of the transactions are fraudulent, a model can achieve very high Accuracy while still failing to detect fraud.

### 5. The best model depends on the metric

Logistic Regression had the highest Recall in cross-validation, while KNN had the highest cross-validation F1-score.

However, the Neural Network achieved the highest F1-score on the final test set.

---

## 8. Limitations and Possible Improvements

There are some parts of the experiments that could be improved.

### Threshold Experiment

The threshold experiment was performed using KNN, while the final model was the Neural Network.

In a future version, I would test thresholds such as 0.2, 0.3, 0.4, 0.5, 0.6 and 0.7 specifically for the Neural Network.

### Class Imbalance

The dataset is highly imbalanced.

I did not use techniques such as SMOTE in this version of the project.

Testing oversampling or other imbalance-handling methods could be useful.

### More Models

More advanced models such as Random Forest, XGBoost or LightGBM could also be tested.

### Hyperparameter Tuning

Only a small number of K values were tested.

A larger search could be performed using techniques such as GridSearchCV or randomized search.

---

## 9. Final Conclusion

The experiments helped me understand how different machine learning decisions affect fraud detection performance.

The most important observations were:

- Feature scaling is very important for KNN.
- The value of k  affects the performance of KNN.
- Classification threshold changes the balance between Precision and Recall.
- Accuracy is not a reliable primary metric for this highly imbalanced dataset.
- Cross-validation can give slightly different results from the final test set.
- The Neural Network achieved the highest F1-score on the final test set.

The final selected model was the **Neural Network (MLP)** with a test F1-score of **0.8333**.

Overall, these experiments helped me better understand the relationship between model selection, preprocessing, evaluation metrics, and the requirements of a real fraud detection problem.