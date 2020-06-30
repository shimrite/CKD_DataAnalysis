from ckdClassifer import *

# This File holds the main method running the classification pipeline
# in order to run the following parameters required:
# - input_data_path - csv file location
# - feature_selection_run_mode - 1 / 2 / 3 [kbest/randomforest/testboth]
# - classifier_run_mode - 1 /2 /3 [knn/randomforest/svm]

def main(input_data_path, feature_selection_run_mode, classifier_run_mode):
    # Set warning flag OFF (these warnings tested and not needed on runtime mode)
    pd.set_option('mode.chained_assignment', None)  # possibly invalid assignment. There may be false positives; situations where a chained assignment is inadvertently reported.
    # Init DataLoader
    dl = DataLoader(input_data_path)
    # PreProcess (clean, impute, scale, handle outliers and correlated features)
    dl.load_and_preprocess_data()
    # FeatureSelection
    fs = FeatureSelectionRun(feature_selection_run_mode, dl)      # init using the Data loaded
    fs.run_fs_model()
    # Classifier        (TBD - cross validation using sub-sampling)
    crf = ClassifierRun(classifier_run_mode, dl, fs)        # init using the Data loaded and the selected features
    crf.classify_by_mode()

    return 0


if __name__ == '__main__':
    try:
        input_data_path = sys.argv[1]
        feature_selection_run_mode = sys.argv[2]
        classifier_run_mode = sys.argv[3]

        status = main(input_data_path, feature_selection_run_mode, classifier_run_mode)
        sys.exit(status)
    except KeyboardInterrupt:
        print("")
        print("DA terminated.")
        sys.exit(1)
