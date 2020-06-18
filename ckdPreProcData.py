import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ckdEDA import *


class DataLoaderParams(object):

    def __init__(self):
        super(DataLoaderParams, self).__init__()
        self.p_test_size = 0
        self.p_random_state = 0
        self.p_na_thresh = 0

    def init_params(self):
        self.p_test_size = 0.33     # train-test split
        self.p_random_state = 48    # train-test split
        self.p_na_thresh = 0.75     # removal features with NA val > thresh
        return

    def update_params(self, tst_sz, rnd_st, na_thrsh):
        self.p_test_size = tst_sz
        self.p_random_state = rnd_st
        self.p_na_thresh = na_thrsh
        return


class DataLoader(object):
    # this class hold the following pre-processing steps:
    # 1. load data and initial exploration
    # 2. split into train and test sets, features(x) and labels(y)
    # 3. data imputation (clean values, NA values)
    # 4. data exploration - outliers detection & handling
    # 5. data scaling
    # 5. data exploration - features statistics, feature correlation, visualization
    # 6. correlated features handling (removal vs pca)

    def __init__(self, data_input_path):
        super(DataLoader, self).__init__()
        self.input_path = data_input_path
        self.df_in = []
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.x_train_valid = []     # data after duplicates removal, NA removal and imputation
        self.x_test_valid = []
        self.params = DataLoaderParams()
        self.params.init_params()       # default values
        self.valid_columns = []     # features remained after NA removal
        self.cat_col_list = []      # categorical features
        self.cat_values_dict = {}   # map between categorical feature and its top value (most common)
        self.mean_num_values = []   # map between numerical feature and its mean value
        self.cat2num_dict = {}      # map between categorical values to numeric labeling (one-hot)
        self.num_col_list = []      # numerical features
        self.y_cat2num_dict = {}    # labeling (one-hot)

    def input_2_train_test(self):
        # this method load the data set and splits the data into train and test sets
        self.df_in = pd.read_csv(self.input_path, index_col=0) #
        print("The Dataset is of shape: " + str(self.df_in.shape))
        print("The Dataset features are: ")
        print(self.df_in.columns)
        self.df_in.info()
        self.df_in.drop_duplicates(inplace=True)
        print("After duplicates removal the Dataset is of shape: " + str(self.df_in.shape))

        # extract labels col
        y_data = self.df_in['classification']
        x_data = self.df_in.drop(columns=['classification'])
        # split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data, test_size=self.params.p_test_size, random_state=self.params.p_random_state)
        print("The train data is of shape: " + str(self.x_train.shape))

    def impute_data(self):
        # -- handle train data --
        # 1. drop features (columns) with NA > 75%
        x_train_val = self.x_train.dropna(axis=1, thresh= self.x_train.shape[0] * self.params.p_na_thresh)
        print("train data set shape after NA droping: " + str(x_train_val.shape))
        self.valid_columns = x_train_val.columns
        # 2. CATEGORICAL val - clean & impute
        # remove "typo" characters
        # create categorical common values dictionary - cat_values_dict - per att its most common value (top)
        # fill NA values with the dictionary values
        self.cat_col_list = x_train_val.select_dtypes(include=np.object).columns.tolist()
        for key in self.cat_col_list:
            tmp = pd.to_numeric(x_train_val.loc[:, key].str.strip().str.strip('?'), errors='ignore')
            x_train_val.loc[:, key] = tmp
            if x_train_val.loc[:, key].dtype == object:
                self.cat_values_dict[key] = x_train_val.loc[:, key].describe().top
        self.cat_col_list = x_train_val.select_dtypes(include=np.object).columns.tolist() # after removal of numerical columns
        x_train_val.fillna(value=self.cat_values_dict, inplace=True)
        # 3. NUMERICAL val - impute  (mean/median/corr)
        num_desc = x_train_val.describe() # note, this is executed only on numerical data
        self.mean_num_values = num_desc.loc['mean']
        x_train_val.fillna(value=self.mean_num_values, inplace=True)
        self.x_train_valid = x_train_val
        # -- handle test data -- get only valid columns, fill NA, scale/fit
        x_test_val = self.x_test[self.valid_columns]
        # clean data
        cat_col_list_test = x_test_val.select_dtypes(include=np.object).columns.tolist()
        for key in cat_col_list_test:
            tmp = pd.to_numeric(x_test_val.loc[:, key].str.strip().str.strip('?'), errors='ignore')
            x_test_val.loc[:, key] = tmp
        cat_col_list_test = x_test_val.select_dtypes(include=np.object).columns.tolist()  # after removal of numeric
        assert self.cat_col_list == cat_col_list_test, "categoriacl columns are differ between train and test sets!"
        # fill NA
        x_test_val.fillna(value=self.mean_num_values, inplace=True)
        x_test_val.fillna(value=self.cat_values_dict, inplace=True)
        self.x_test_valid = x_test_val
        return

    def scale_data(self):
        # scale train data
        # NUMERICAL features scaling by Min-Max scaling
        mms = MinMaxScaler() # fit numerical data
        self.num_col_list = self.x_train_valid.select_dtypes(include=np.number).columns.tolist()
        self.x_train_valid.loc[:, self.num_col_list] = mms.fit_transform(self.x_train_valid.loc[:, self.num_col_list])
        # CATEGORICAL features scaling
        self.cat2num_dict = {}
        for key in self.cat_col_list: # create categorical values dictionary - per att its optional values and their numerical replacement (top)
            self.cat2num_dict[key] = {self.x_train_valid.loc[:, key].value_counts().keys()[0]: 0, self.x_train_valid.loc[:, key].value_counts().keys()[1]: 1}
        self.x_train_valid.replace(self.cat2num_dict, inplace=True) # scale categorical data (TBD - onehot!)

        # scale Test data
        self.x_test_valid.replace(self.cat2num_dict, inplace=True)
        self.x_test_valid.loc[:, self.num_col_list] = mms.fit_transform(self.x_test_valid.loc[:, self.num_col_list])

        # scale labels to 0/1
        # for key in y_train.keys():
        self.y_train = self.y_train.str.strip()
        self.y_test = self.y_test.str.strip()
        self.y_cat2num_dict = {self.y_train.value_counts().keys()[0]: 0, self.y_train.value_counts().keys()[1]: 1}
        self.y_train.replace(self.y_cat2num_dict, inplace=True)
        self.y_test.replace(self.y_cat2num_dict, inplace=True)

    def drop_features(self, col_2b_dropped):
        self.x_train_valid.drop(columns=col_2b_dropped)
        self.x_test_valid.drop(columns=col_2b_dropped)

    def load_and_preprocess_data(self):
        self.input_2_train_test()
        self.impute_data()
        # self.scale_data()
        ed = EDA(self.x_train_valid)
        ed.info_data()
        ed.statistics_data()
        ed.outliers_detection()
        self.drop_features(ed.outliers_col_2b_dropped)
        self.scale_data()
        ed2 = EDA(self.x_train_valid)
        ed2.statistics_data()
        ed2.correlated_features_detection()
        self.drop_features(ed.corr_col_2b_dropped)


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

    return 0


if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except KeyboardInterrupt:
        print("")
        print("DA terminated.")
        sys.exit(1)
