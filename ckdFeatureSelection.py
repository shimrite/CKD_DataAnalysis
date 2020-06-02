from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from ckdPreProcData import *

# this section holds the following:
# 1. K best by chi^2 statistic model
# 2. Random Forest model
# 3. models comparison


class FeatureSelectionParams(object):

    def __init__(self):
        super(FeatureSelectionParams, self).__init__()
        self.k = 10


class FeatureSelectionRun(object):

    def __init__(self, input_mode, in_data: DataLoader):
        super(FeatureSelectionRun, self).__init__()
        self.run_mode = input_mode  # 1 / 2 / 3 [kbest/randomforest/testboth]
        self.x_train = in_data.x_train_valid
        self.x_test = in_data.x_test_valid
        self.y_train = in_data.y_train
        self.y_test = in_data.y_test
        self.col_names = self.x_train.columns.values
        self.params = FeatureSelectionParams()
        self.selected_features_indexes = []
        self.selected_features_names = []

    def run_kbest_model(self):
        # K best by chi^2 statistic model
        model = SelectKBest(score_func=chi2, k=self.params.k)
        fit = model.fit(self.x_train, self.y_train)
        # summarize scores
        np.set_printoptions(precision=3)
        print(fit.scores_)
        features = fit.transform(self.x_train)
        # summarize selected features
        #print(features[0:15, :])  # print data of the K best features
        print("--- K Best Features ---")
        print("Features Indx sorted by their score:")
        print(np.argsort(fit.scores_)[-self.params.k:])  # best K features indexes (10)
        # print("Best Features Data:")
        # print(self.x_train.values[:15, np.argsort(fit.scores_)[-self.params.k:]])  # print data of the K best features
        print("Features Names sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), fit.scores_), self.col_names), reverse=True))
        # update features selected
        self.selected_features_indexes = np.argsort(fit.scores_)[-self.params.k:]
        self.selected_features_names = sorted(zip(map(lambda x: round(x, 4), fit.scores_), self.col_names), reverse=True)

    def run_random_forest_model(self):

        # Random Forest model
        model = RandomForestClassifier()
        # Fit the model
        model.fit(self.x_train, self.y_train)
        # summarize selected features
        print("--- Random Forest Features ---")
        print("Features Indx sorted by their score:")
        print(np.argsort(model.feature_importances_)[-self.params.k:])  # best K features indexes (10)
        print("Features Names sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), self.col_names), reverse=True))
        # update features selected
        self.selected_features_indexes = np.argsort(model.feature_importances_)[-self.params.k:]
        self.selected_features_names = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), self.col_names), reverse=True)



