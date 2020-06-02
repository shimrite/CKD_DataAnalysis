import sys
from ckdPreProcData import *
from ckdFeatureSelection import *


def main():
    # run mode config (runtime/test)
    # these warnings tested and not needed on runtime mode
    pd.set_option('mode.chained_assignment', None)  # possibly invalid assignment. There may be false positives; situations where a chained assignment is inadvertently reported.
    # set input file
    input_data = '/Users/shimriteliezer/Documents/testProjects/CKDanalysis/CKD.csv'
    # init DataLoader
    dl = DataLoader(input_data)
    # preProcess
    dl.load_and_preprocess_data()
    # init FeatureSelection
    run_mode = 1
    fs = FeatureSelectionRun(run_mode, dl)
    fs.run_kbest_model()
    fs.run_random_forest_model()

    return 0


if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except KeyboardInterrupt:
        print("")
        print("DA terminated.")
        sys.exit(1)
