import os
import joblib
import json
from data_prep import (
    Load_data,
    dataset_info,
    CheckClassDistribution,
    CheckFeature,
    CheckFeatureAmount,
    CheckFeatureTime,
    CheckMissingValue,
    prepare_data
)
from train import (
    train_logistic_regression,
    train_knn,
    train_decision_tree,
    evaluate_model,
    cross_validate_model,
    compare_hyperparameters_KNN,
    compare_thresholds,
    save_model,
    train_neural_network_sklearn,
    evaluate_neural_network ,
    cross_validate_neural_network
)
from predict import load_model, load_scaler, predict


# ============================================
# آموزش و آزمایش‌ها
# ============================================
def main():

    print("CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")

    
    # 1. بارگذاری داده
    print("\n[1] Loading data")
    df = Load_data()
    print("Data loaded: " + str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns")
    
    # 2. بررسی داده
    print("\n[2] Data Analysis")
    dataset_info(df)
    class_dist = CheckClassDistribution(df)
    print("\nClass Distribution:")
    print("   Legitimate (0): " + str(class_dist[0]) + "%")
    print("   Fraud (1):     " + str(class_dist[1]) + "%")
    
    CheckFeature(df)
    CheckFeatureAmount(df)
    CheckFeatureTime(df)
    CheckMissingValue(df)
    
    # 3. آماده‌سازی داده
    print("\n[3] Preparing data")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print("Training set: " + str(len(X_train)) + " samples")
    print("Test set:     " + str(len(X_test)) + " samples")
    
    # 4. آموزش مدل‌ها
    print("\n[4] Training models")
    print("   Logistic Regression  ")
    lr_model = train_logistic_regression(X_train, y_train)
    print("   KNN  ")
    knn_model = train_knn(X_train, y_train)
    print("   Decision Tree   ")
    dt_model = train_decision_tree(X_train, y_train)
    print("   Neural Network (MLP)...")
    nn_model= train_neural_network_sklearn(X_train, y_train)
    print("All models trained!")
    
    # 5. ارزیابی
    print("\n[5] Evaluating models...")

    models = {
        "Logistic Regression": lr_model,
        "KNN": knn_model,
        "Decision Tree": dt_model,
        "Neural Network": nn_model
    }

    results = {}

    print("\n   Model | Accuracy | Precision | Recall | F1-Score")
    
    for name, model in models.items():
       if name == "Neural Network":
           nn_results = evaluate_neural_network(nn_model, X_test, y_test)
           results[name] = nn_results
           print("   " + name + " | " + str(nn_results['accuracy']) + " | " + str(nn_results['precision']) + " | " + str(nn_results['recall']) + " | " + str(nn_results['f1']))
           print("   Confusion Matrix: " + str(nn_results['confusion_matrix']))
       else:
           eval_results = evaluate_model(model, X_test, y_test)
           results[name] = eval_results
           print("   " + name + " | " + str(eval_results['accuracy']) + " | " + str(eval_results['precision']) + " | " + str(eval_results['recall']) + " | " + str(eval_results['f1']))
           print("   Confusion Matrix: " + str(eval_results['confusion_matrix']))
    # 6. Cross Validation
    print("\n[6] Cross Validation (5-Fold)")

    print("\n   Model | Precision | Recall | F1-Score")

    for name, model in models.items():
       if name == "Neural Network":
           cv_nn_results = cross_validate_neural_network(X_train, y_train)
           print("   " + name + " | " + str(cv_nn_results['mean_precision']) + " | " + str(cv_nn_results['mean_recall']) + " | " + str(cv_nn_results['mean_f1']))
       else:
           cv_scores = cross_validate_model(model, X_train, y_train)
           print("   " + name + " | " + str(cv_scores['mean_precision']) + " | " + str(cv_scores['mean_recall']) + " | " + str(cv_scores['mean_f1']))
  
    print("\n[7] KNN with/without scaling")

    # KNN بدون Scaling
    from sklearn.neighbors import KNeighborsClassifier
    knn_no_scaler = KNeighborsClassifier(n_neighbors=5)
    knn_no_scaler.fit(X_train, y_train)
    results_no_scaler = evaluate_model(knn_no_scaler, X_test, y_test)

    # KNN با Scaling (همون مدل قبلی)
    results_with_scaler = results["KNN"]

    print("   KNN Without Scaling: Precision=" + str(results_no_scaler['precision']) + ", Recall=" + str(results_no_scaler['recall']) + ", F1=" + str(results_no_scaler['f1']))
    print("   KNN With Scaling:    Precision=" + str(results_with_scaler['precision']) + ", Recall=" + str(results_with_scaler['recall']) + ", F1=" + str(results_with_scaler['f1']))
    # 7. Hyperparameter
    print("\n[8] Hyperparameter Experiment (KNN)")
    knn_params = compare_hyperparameters_KNN(X_train, X_test, y_train, y_test)
    print("\n   K Value | Precision | Recall | F1-Score")
    for res in knn_params:
       print("   K=" + str(res['k']) + " | " + str(res['precision']) + " | " + str(res['recall']) + " | " + str(res['f1']))
    
    # 9. Threshold
    print("\n[9] Threshold Experiment")
    thresholds = compare_thresholds(knn_model, X_test, y_test)
    print("\n   Threshold | Precision | Recall | F1-Score")
    for res in thresholds:
        print("   " + str(res['threshold']) + " | " + str(res['precision']) + " | " + str(res['recall']) + " | " + str(res['f1']))
    
    # 10. انتخاب بهترین
    print("\n[10] Selecting best model...")

    # انتخاب از بین همه مدل‌ها)
    best_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_name]
    best_f1 = results[best_name]['f1']
    best_threshold = max(thresholds, key=lambda x: x['f1'])['threshold']

    print("   Best Model: " + best_name)
    print("   F1-Score: " + str(best_f1))
    print("   Best Threshold: " + str(best_threshold))
    
    # 11. ذخیره مدل
    print("\n[11] Saving model...")
    os.makedirs("models", exist_ok=True)

    if best_name == "Neural Network":
       joblib.dump(best_model, "models/nn_model.pkl")
       print("   Neural Network saved to models/nn_model.pkl")
    else:
       save_model(best_model, "models/model.pkl")
       print("   Model saved to models/model.pkl")
    
       if best_name == "KNN":
          scaler_obj = best_model.named_steps['scaler']
          joblib.dump(scaler_obj, "models/scaler.pkl")
          print("   Scaler saved to models/scaler.pkl")
    
    # گزارش نهایی
    print("FINAL REPORT")
    print("   Best Model:              " + best_name)
    print("   Test F1-Score:           " + str(best_f1))
    print("   Test Recall:             " + str(results[best_name]['recall']))
    print("   Test Precision:          " + str(results[best_name]['precision']))
    print("   Selected Threshold:      " + str(best_threshold))
    print("   Confusion Matrix:")
    print("   " + str(results[best_name]['confusion_matrix']))
    print("TRAINING COMPLETE!")



# تست پیش‌بینی

def test_prediction():
    
    print("PREDICTION TEST")
    
    # نمونه داده ورودی 
    df = Load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    import random
    random_index = random.randint(0, len(X_test) - 1)
    sample_input = X_test.iloc[random_index].to_dict()
    
    try:
        # 1. بارگذاری مدل
        print("\n[1] Loading model")
        model = load_model()
        print("Model loaded successfully")
        
        # 2. بارگذاری اسکیلر
        print("\n[2] Loading scaler")
        scaler = load_scaler()
        if scaler is not None:
            print(" Scaler loaded successfully")
        else:
            print(" No scaler found, proceeding without scaling")
        
        # 3. انجام پیش‌بینی
        print("\n[3] Making prediction")
        result = predict(model, sample_input, scaler, threshold=0.5)
        
        # 4. نمایش نتیجه
        print("\n[4] Result:")
        print(json.dumps(result, indent=2))
        
        # 5. تفسیر نتیجه
        print("\n[5] Interpretation:")
        print("    Transaction is: " + result['prediction'])
        print("    Probability of fraud: " + str(result['probability'] * 100) + "%")
        print("    Threshold used: " + str(result['threshold']))
        print("    Status: " + result['status'])
        
    except FileNotFoundError as e:
        print("\n Error: " + str(e))
        print("    Please train the model first by running: python main.py")
    except Exception as e:
        print("\n Unexpected error: " + str(e))


# اجرا
if __name__ == "__main__":
    # اجرای آموزش و آزمایش‌ها
    main()
    
    print("RUNNING PREDICTION TEST")
    test_prediction()