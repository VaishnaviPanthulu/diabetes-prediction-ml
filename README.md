# Diabetes Prediction using Machine Learning

This project focuses on predicting diabetes using various machine learning algorithms across **three different datasets**:  
1. PIMA Indian Diabetes Dataset  
2. NHANES 2013–2014  
3. Diabetes 130-US Hospitals for Years 1999–2008  

The goal is to compare the performance of different models and understand which factors contribute most to diabetes prediction.

---

## Datasets Used

### 1. PIMA Indian Diabetes Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features: Glucose, Blood Pressure, BMI, Age, etc.
- Binary Classification: Diabetic (1) or Not (0)

### 2. NHANES 2013–2014
- Source: CDC NHANES public dataset
- Includes demographics, lab test results, and health questionnaire responses

### 3. Diabetes 130-US Hospitals (1999–2008)
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)
- Large-scale dataset with patient records from hospitals
- Contains readmission rates, medications, diagnoses, etc.

---

## ML Algorithms Implemented

- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  
- Feedforward Neural Network (FNN) (TensorFlow/Keras)


---

## Evaluation Metrics

Each model was evaluated using:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC Curve (for some models)

---

## Results Summary

| Dataset      | Best Model        | Accuracy | Notes |
|--------------|-------------------|----------|-------|
| PIMA         | Random Forest     | ~78%     | Balanced features |
| NHANES       | Logistic Regression | ~75%  | Small sample used |
| 130-US Hosp. | SVM or RF         | ~80–85%  | Requires careful preprocessing |

> Neural Networks slightly improved performance for some datasets.

---

## Requirements

Install dependencies using:

pip install -r requirements.txt
Key libraries:

pandas

scikit-learn

matplotlib, seaborn

tensorflow, keras (for FNN)



## Conclusion

This project demonstrates the effectiveness of machine learning algorithms in predicting diabetes based on real-world health datasets. By working with diverse datasets like PIMA, NHANES, and the 130-US Hospitals records, we explored the strengths and limitations of models like Random Forest, SVM, Logistic Regression, and Neural Networks.

Through this analysis, we gained valuable insights into data preprocessing, model evaluation, and real-world applicability of predictive models in healthcare. The results indicate that with proper feature selection, balanced data, and tuned models, early prediction of diabetes can be significantly improved — supporting preventive healthcare efforts.

This project lays a strong foundation for future work in health analytics and predictive modeling, particularly for chronic disease monitoring and clinical decision support systems.
