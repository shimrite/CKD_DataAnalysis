from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from ckdPreProcData import *


class FeatureSelectionParams(object):
    # This class holds the default parameters
    def __init__(self):
        super(FeatureSelectionParams, self).__init__()
        self.k = 10


class FeatureSelectionRun(object):
    # This Class performs Feature Selection model according the 'run_mode':
    # run_mode=1: K best by chi^2 statistic model
    # run_mode=2: Random Forest model
    # run_mode=3: models merge option
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
        self.selected_features_names = [] # not in use TBD

    def run_fs_model(self):
        # run the feature selection model by run_mode:
        if self.run_mode==1 :
            self.run_kbest_model()
        elif self.run_mode==2 :
            self.run_random_forest_model()
        else:
            self.run_kbest_model()
            kbf = self.selected_features_indexes
            self.run_random_forest_model()
            rff = self.selected_features_indexes
            self.merge_best_features(kbf, rff, self.params.k)

    def merge_best_features(self, a, b, n):
        # This is a naive merge:
        # 1. Take the intersection set
        # 2. If the intersection set is less than n - take the best 3 features of each set
        # TBD - merge by features scores, similarity, strength
        intersect = set(a).intersection(b)
        next_best_features = n-len(intersect)
        if next_best_features > 0 :
            a_cont = [value for value in a if value not in intersect]
            b_cont = [value for value in b if value not in intersect]
            self.selected_features_indexes = intersect + a_cont[:int(next_best_features/2)] + b_cont[:int(next_best_features/2)]
        else:
            self.selected_features_indexes = intersect

    def run_kbest_model(self):
        # K best by chi^2 statistic model
        model = SelectKBest(score_func=chi2, k=self.params.k)
        model_results = model.fit(self.x_train, self.y_train)
        # Summarize scores
        np.set_printoptions(precision=3)
        print(model_results.scores_)
        features = model_results.transform(self.x_train)
        # Summarize selected features
        # print(features[0:15, :])  # print data of the K best features
        print("--- K Best Features ---")
        print("Features Indx sorted by their score:")
        print(np.argsort(model_results.scores_)[-self.params.k:])  # best K features indexes (10)
        # print("Best Features Data:")
        # print(self.x_train.values[:15, np.argsort(model_results.scores_)[-self.params.k:]])  # print data of the K best features
        print("Features Names sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 4), model_results.scores_), self.col_names), reverse=True))
        # update features selected
        self.selected_features_indexes = np.argsort(model_results.scores_)[-self.params.k:]
        self.selected_features_names = sorted(zip(map(lambda x: round(x, 4), model_results.scores_), self.col_names), reverse=True)

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



