import sys
from ckdPreProcData import *
from ckdFeatureSelection import *
from ckdClassifer import *


def main():
    # run mode config (runtime/test)
    # these warnings tested and not needed on runtime mode
    pd.set_option('mode.chained_assignment', None)  # possibly invalid assignment. There may be false positives; situations where a chained assignment is inadvertently reported.
    # set input file
    input_data = '/Users/shimriteliezer/Documents/testProjects/CKDanalysis/CKD.csv'
    # init DataLoader
    dl = DataLoader(input_data)
    # preProcess (clean, impute, scale, handle outliers and correlated features)
    dl.load_and_preprocess_data()
    # FeatureSelection
    run_mode = 2
    fs = FeatureSelectionRun(run_mode, dl)      # init using the Data loaded
    fs.run_fs_model()
    # classifier        (TBD - cross validation using sub-sampling)
    run_mode = 3
    crf = ClassifierRun(run_mode, dl, fs)        # init using the Data loaded and the selected features
    crf.classify_by_mode()

    return 0


if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except KeyboardInterrupt:
        print("")
        print("DA terminated.")
        sys.exit(1)
