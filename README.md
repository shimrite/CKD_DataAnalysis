#  CKD_DataAnalysis
Predicting CKD (chronic kidne diseas) based on subject health records.

    * For further details on the CKD dataset and the task please refer to the Data_Analysis_Task.doc file.

The project contains the following files:
### 1. ckdRunDataAnalysisAndPredict.py - 
This File holds the main method running the classification pipeline including: 

Load data and Pre-processing --> Feature selection --> Classification.

In order to run the following parameters required:
- input_data_path - csv file location
- feature_selection_run_mode - 1 / 2 / 3 [kbest/randomforest/testboth]
- classifier_run_mode - 1 /2 /3 [knn/randomforest/svm]

      ckdRunDataAnalysisAndPredict.py '/Users/.../CKDanalysis/CKD.csv' 2 3

### 2. ckdPreProcData.py - 
This file holds the following pre-process steps: 

       1. Load data and initial exploration 
       2. Split into train and test sets, features(x) and labels(y)
       3. Data imputation (clean values, NA values)
       4. Data exploration - outliers detection & handling
       5. Data scaling
       5. Data exploration - features statistics, feature correlation, visualization
       6. Correlated features handling (removal vs pca)
### 3. ckdEDA.py -
This file holds all Exploration Data Analysis methods:

       1. High level info on the data set
       2. Features statistics
       3. Feature correlation
       4. Data scaling
       5. Outliers detection
       6. Visualization
### 3. ckdFeatureSelectionRun.py - 
This file perform feature selection model on the dataset.

    * The feature selection model selected by run_mode parameter: [1/2/3] stands for [KBest/RandomForest/Both] respectevly
### 4. ckdClassifier.py - 
This file holds the following:

       1. KNN classifier
       2. Random Forest classifier
       3. SVM classifier
       4. Classifier Results Analysis (ROC/AUC)
       
    * The classifier model selected by run_mode parameter.
    * The model is performed on the preprocessed data using the selected features from previous steps.
