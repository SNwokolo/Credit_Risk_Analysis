# Credit_Risk_Analysis
Applying machine learning models and methods in predicticting credit risk.

## Overview of analysis
The purpose of this analysis is to implement machine learning techniques in solving credit card risk problems. Methods such as over- and undersampling are employed along with scikit and other  libraries in Python to build and evaluate several machine learning models to predict credit risk. Predicting credit risk can assist banks and financial institutions foresee anomalies, reduce risk cases, provide recommendations on what to do in cases of fraud. The analysis also evaluates the performance of the machine learning models to provide direction on whether they should be used to predit credit risk.

## Results

To test our predictions using undersamplimg and oversampling methods, the following models were implemented:

- The random oversampling model - the balanced accuracy score obtained was 0.62. The precision for high risk was 0.01 while that of low risk was 1.00. The recall for high risk was 0.58 and the low risk score was 0.65. 
![Random Oversampling](https://github.com/SNwokolo/Credit_Risk_Analysis/blob/bbf6d28bab9526aa7398e2b6f036dba7a9056d44/Images/random%20oversampling.png)

- The SMOTE model - 0.62 was the balanced accuracy score obtained. The precision score for high risk applications was 0.01 and that of low risk was 1.00. The recall scores were 0.6 and 0.65 for high risk and low risk respectively. ![SMOTE](https://github.com/SNwokolo/Credit_Risk_Analysis/blob/bbf6d28bab9526aa7398e2b6f036dba7a9056d44/Images/SMOTE.png)

- The cluster centroids undersampling model - the balanced accuracy score obtained was 0.51. The precision for high risk was 0.01 while that of low risk was 0.99. The recall for high risk was 0.59 and the low risk recall score was 0.42. 
![Cluster Centroids](https://github.com/SNwokolo/Credit_Risk_Analysis/blob/bbf6d28bab9526aa7398e2b6f036dba7a9056d44/Images/Undersampling.png)


- The SMOTEENN (combines over- and undersampling) - the balanced accuracy score obtained was 0.64. The precision for high risk was 0.01 while that of low risk was 1.00. The recall for high risk was 0.72 and the low risk recall score was 0.57. ![SMOTEENN](https://github.com/SNwokolo/Credit_Risk_Analysis/blob/3278d7ea7c7bafcd28950e19bf6e7cb72c1a3dce/Images/SMOTEENN.png)

For ensemble learner algorithms, we used the following models:

- Random Forest Classifier - the balanced accuracy score obtained was 0.66. The precision for high risk was 0.59 while that of low risk was 1.00. The recall for high risk was 0.32 and the low risk recall score was 1.00. 
![Random Forest Classifier](https://github.com/SNwokolo/Credit_Risk_Analysis/blob/bbf6d28bab9526aa7398e2b6f036dba7a9056d44/Images/RFC.png)

- Easy Ensemble model - the balanced accuracy score obtained was 0.93. The precision for high risk was 0.07 while that of low risk was 1.00. The recall for high risk was 0.92 and the low risk recall score was 0.93. ![Easy Ensemble](https://github.com/SNwokolo/Credit_Risk_Analysis/blob/bbf6d28bab9526aa7398e2b6f036dba7a9056d44/Images/Easy%20Ensemble.png)


## Summary
The model with the lowest balanced accuracy score was found to be the cluster centroids model followed by the SMOTE then SMOTEENN then random oversampling, then the random forest classifier and finally the highest score went to the easy ensemble model. A lower balanced accuracy score is an indication that the model is not very good at performing predictions. 
Based on the results obtained from each analysis, the best model out if the six for predicting credit risk would be the Easy Ensemble model. The reason for this is the fact that it had the highest balanced accuracy score of all six models. This model also had higher recall scores for both high risk and low risk credit applications meaning that there are fewer false negatives observed in the predicted values. Although the f1 score of the high risk application is rather low at 0.13 which means that there's an imbalance between the precision and recall scores, the higher value for the low risk means that the model is  better at predicting those credit applications that are lower risk. The precision was observed to be low for the high risk implying that the there is a large number of false positives which is not reliable positive classification for applications that are high risk.
