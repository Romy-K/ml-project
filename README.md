# Machine learning project: Predicting the Eurovision 2025 Grand Final top 10

Using several machine learning models to try and predict the top 10 countries in the 2025 Grand Final. 
I used data from 2016-2024 to train my models, then applied the models to the actual 2025 data.

## Data used 
The dataset eurovision_2016-25, found on Kaggle.   
link: https://www.kaggle.com/datasets/rhyspeploe/eurovision-2016-25?resource=download

## Models used
- KNN
- KNN + GridSearch and SMOTE
- RandomForest

## Outcomes 
The confusion matrix outcomes of the 3 models, applies to the 2025 data, are:
- KNN: 7 TP, 1 FP, 26 TN, 3 FN
- KNN +: 10 TP, 5 FP, 0 FN, 22 TN
- RandomForest: 10 TP, 0 FP, 0 FN, 27 TN

RandomForest managed to predict the 2025 top 10 perfectly. 

## Libraries used
pandas    
numpy   
seaborn   
matplotlib.pyplot as plt   
sklearn.model_selection: train_test_split, GridSearchCV   
sklearn.preprocessing: StandardScaler   
sklearn.neighbors: KNeighborsClassifier   
sklearn.metrics: classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score    
imblearn.over_sampling: SMOTE   
sklearn.ensemble: RandomForestClassifier   
mblearn.pipeline: Pipeline   

## Contact
Romy Koreman   
romykoreman@live.nl