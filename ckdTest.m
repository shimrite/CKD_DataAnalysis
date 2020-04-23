function [train, test, scaledData, validFeaturesList, validFeaturesSum, outliersIndPerFeature]= preproc_data(file_path)

% -load data-
% init validFeaturesLis = all features

% -handle missing data-
% find missing per att -> features with missing% > 10% ->remove from validFeaturesList
% find outliers -> val exceed min/max, val exceed feature distribution (e.g. value is > 2std from avg)
% add indx to outliersIndPerFeature ("point of interest" may be used during feature selection model)
% get avg&std per att (on existing values without ouliers values) ->  fill avg / predict expected value by strongest features
% *strongest features - no missing values

% -scale att values-
% numerical -> val-avg/std, nominal -> replace val to numerical labels 

% -traim-test split- (done on scaled data)
% count by label > 150/250
% get equale numbers of subjects (healthy-ckd)
% rand 60% ind from 150 per label -> train

% validFeaturesSum = list of characters per feature: n, avg, std,
% missingPercentile, min, nax

function [classifier, featuresList] = ckdModel( scaledData, validFeaturesList, validFeaturesSum, outliersIndPerFeature)

% train test split

% - feature selection - 
% bagging - diffferent trains done in paralel on random subset of features 
% vs 
% boosting - train on all features (e.g. each iteration select "minPaneltyAtt" and add to featuresList)    

% - classifier -
% SVM
% vs
% Regresssion Trees
