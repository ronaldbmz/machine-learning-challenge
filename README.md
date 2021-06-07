# machine-learning-Exoplanet Exploration

## Machine Learning Homework - Exoplanet Exploration

## Data

### Context
The Kepler Space Observatory is a NASA-build satellite that was launched in 2009. The telescope is dedicated to searching for exoplanets in star systems besides our own, with the ultimate goal of possibly finding other habitable planets besides our own. The original mission ended in 2013 due to mechanical failures, but the telescope has nevertheless been functional since 2014 on a "K2" extended mission.
Kepler had verified 1284 new exoplanets as of May 2016. As of October 2017, there are over 3000 confirmed exoplanets total (using all detection methods, including ground-based ones). The telescope is still active and continues to collect new data on its extended mission.

### Content
The dataset contains total 41 columns including the target variable and total 6991 records. Target variable is “koi_disposition” which contains following values:
•	CONFIRMED – with total 1800 records and it is one of the predicting values.
•	FALSE POSITIVE – with total 3504 records and it is also one of the predicting values.
•	CANDIDATE – with total 1687 records and is required to be used as new dataset to test the predictability of the built model.

### Data Analysis
Following steps are being taken as a part of the data analysis:

•	Data Info - To check all the columns and their types. Using data info we can notice that 3 attributes are grouped together i.e. one is the actual attribute value and other two are that variable’s error value. 
For example, there is attribute “koi_period” and other two associated attributes are “koi_period_err1” and “koi_period_err2”.
•	Statistical Summary – This is just to check the mean, count, min and max of each variable to get a high-level view.
•	Correlation Coefficient – This is to check the correlation between different variables so that in the later feature selection step we can remove highly correlated variables. With this analysis, we can observe that the “error” variables are highly correlated with its respective variable

### Data Cleaning
Checked for missing, NULL and NA values in the dataset and found that in the given data there were no missing, NULL or NA values.

### Feature Selection for Model
As observed from correlation coefficient analysis, the “error” variables are highly correlated to its respective variable therefore, in this feature selection step removed variables like 'koi_teq_err1', 'koi_teq_err2' etc.

### Train Test Split
Given data has been split into following segments:
•	Post model test split – The data has been filtered for type “CANDIDATE” which is needed to be used for final prediction once the model gets finalized. 
Further this data itself has been divided into “X_candidates” where it only contains the predictor variables.
•	Standard train and test split - Remaining data has been divided into standard X_train, X_test, y_train, y_test, where train contains 60% and test 40% of the data. “y_train” and “y_test” contains the prediction values i.e. “CONFIRMED” and “FALSE POSITIVE”.

### Pre-processing
•	Scale the “train” data using the MinMaxScaler and transform all the datasets – “X_train”, “X_test” and “X_candidates” using the scaler generated from scaling “train” data
•	Converted target variable in one-hot encoding with “FALSE POSITIVE” as 0 and “CONFIRMED” ass 1.

## Models
### Model-1: Random Forest Classifier
First model is Random Forest with n_estimators=200 and got following result:
•	Training data score – 1.0
•	Testing data score – 0.986
•	Precision – FALSE POSITIVE: 0.98 and CONFIRMED: 0.99
•	F-1 Score – FALSE POSITIVE: 0.99 and CONFIRMED: 0.98

Also, from the feature importance result we can observe that all the selected predictor variables are being used in the modelling. There is no variable with “0” importance value.

### Model-2: KNN Classifier
Next model is KNN. To determine the best value of k in the model, we loop through different k values to see which has the highest accuracy. We only use odd numbers because we don't want any ties. After the model building, we can confirm that the best value of k is 13 with following result:
•	Training data score – 0.982
•	Testing data score – 0.982
•	Precision – FALSE POSITIVE: 0.99 and CONFIRMED: 0.97
•	F-1 Score – FALSE POSITIVE: 0.99 and CONFIRMED: 0.97

Comparison: Compared to Random Forest, KNN model’s test score is less and the F-1 Score for both target variable’s values is also lower. Therefore, we can confirm that Random Forest model is better than KNN.

### Model-3: SVM Classifier
Third model is SVM classifier with “linear” kernel. In this version of model building all the model parameters have been set to default. Following is the result from SVM model:
•	Training data score – 0.989
•	Testing data score – 0.991
•	Precision – FALSE POSITIVE: 0.99 and CONFIRMED: 1.00
•	F-1 Score – FALSE POSITIVE: 0.99 and CONFIRMED: 0.99

Comparison: Compared to Random Forest and KNN, this SVM model is the best with the highest training and test score along with Precision and F-1 score.

### Model-4: SVM Classifier with parameter tuning with Grid Search
The last model is the same as 3rd model i.e. SVM classifier with “linear” kernel. But, in this version we performed parameter tuning using the Grid Search to get the best possible parameters for model building. Following is the result from the grid search and model:
•	Grid best parameters - {'C': 1, 'gamma': 0.0001}
•	Training data score – 0.989
•	Testing data score – 0.991
•	Precision – FALSE POSITIVE: 0.99 and CONFIRMED: 1.00
•	F-1 Score – FALSE POSITIVE: 0.99 and CONFIRMED: 0.99

Comparison: Compared to 3rd model (SVM classifier without parameter tuning), we observed that there is no significant improvement in the model performance in the model with parameter tuning. Therefore, in this case we can go with the 3rd model i.e. SVM classifier without parameter tuning.

## Final Saved Model
From results of all the 4 models, SVM classifier is the best performing model with the highest training and test score accuracy and other metrics like precision and f-1score.
 The final version of SVM model has been saved to file named “koi_disposition_svm_model.sav” using “joblib” library.

