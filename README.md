#  CKD_DataAnalysis
Predicting CKD (chronic kidne diseas) based on subject health records.
* for further details on the CKD dataset and the task please refer to the Data_Analysis_Task.doc file.

The project contains the following files:
### 1. ckdRunDataAnalysisAndPredict.py - 
the main "run", in order to run this file please update the input_data path.
this file runs all steps of classification pipeline: 

load and pre-process the data, feature selection, classification.
### 2. ckdPreProcData.py - 
this file holds the following pre-process steps: 

    1. load data and initial exploration 
    2. split into train and test sets, features(x) and labels(y)
    3. data imputation (clean values, NA values)
    4. data exploration - outliers detection & handling
    5. data scaling
    5. data exploration - features statistics, feature correlation, visualization
    6. correlated features handling (removal vs pca)
### 3. ckdEDA.py -
this file holds all Exploration Data Analysis methods:

    1. high level info on the data set
    2. features statistics
    3. feature correlation
    4. data scaling
    5. outliers detection
    6. Visualization
### 3. ckdFeatureSelectionRun.py - 
this file perform feature selection model on the dataset.

the model selected by run_mode parameter: [1/2/3] stands for [KBest/RandomForest/Both] respectevly
### 4. ckdClassifier.py - 
this file holds the following:

    1. KNN classifier
    2. Random Forest classifier
    3. SVM classifier
    4. classifier analysis (ROC/AUC)

    the model selected by run_mode parameter.
    the model is performed on the preprocessed data usin the selected features from previous steps.
