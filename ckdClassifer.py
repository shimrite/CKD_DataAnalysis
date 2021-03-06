from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.neighbors import KNeighborsClassifier
from ckdFeatureSelection import *
import matplotlib.pyplot as plt


class ClassifierParams(object):
    # This class holds the default parameters
    def __init__(self):
        super(ClassifierParams, self).__init__()
        self.k = 10
    def update_params(self, k_new):
        self.k = k_new
        return


class ClassifierRun(object):
    # This class perform the following classification models:
    # 1. KNN classifier
    # 2. Random Forest classifier
    # 3. SVM classifier
    # Also, it analysis (ROC/AUC)
    def __init__(self, run_mode, in_data: DataLoader, in_features: FeatureSelectionRun):
        super(ClassifierRun, self).__init__()
        self.run_mode = run_mode  # 1 / 2 / 3 [knn/randomforest/svm]
        self.x_train = in_data.x_train_valid
        self.x_test = in_data.x_test_valid
        self.y_train = in_data.y_train
        self.y_test = in_data.y_test
        self.col_names = self.x_train.columns.values
        self.params = ClassifierParams()
        self.selected_features_indexes = in_features.selected_features_indexes
        self.selected_features_names = in_features.selected_features_names
        self.x_train = self.x_train.iloc[:, self.selected_features_indexes]
        self.x_test = self.x_test.iloc[:, self.selected_features_indexes]
        self.model = []

    def classify_by_mode(self):
        if self.run_mode==1 :
            self.fit_KNN_model()
        elif self.run_mode==2 :
            self.fit_random_forest_model()
        else:
            self.fit_svm_model()
        self.classify_model()

    def fit_KNN_model(self):
        # K Nearest Neighbours model
        self.model = KNeighborsClassifier(n_neighbors=self.params.k)
        # Fit the model
        self.model.fit(self.x_train, self.y_train)

    def fit_random_forest_model(self):
        # Random Forest model
        self.model = RandomForestClassifier()
        # Fit the model
        self.model.fit(self.x_train, self.y_train)

    def fit_svm_model(self):
        self.model = svm.SVC(kernel='linear', C=1.0, probability=True)        # TBD, check data balancing, update class_weight={1: 10})
        self.model.fit(self.x_train, self.y_train)
        # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

    def classify_model(self):   #by_random_forest

        preds = self.model.predict(self.x_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test.values, preds, pos_label=2)

        preds_prob_pos = self.model.predict_proba(self.x_test)[:, 1]

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(self.y_test.values, preds_prob_pos, n_bins=10)

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % ("RF",))

        ax2.hist(preds_prob_pos, range=(0, 1), bins=10, label="RF",
                 histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()

        print(roc_auc_score(self.y_test.values, preds_prob_pos, multi_class='ovo'))




