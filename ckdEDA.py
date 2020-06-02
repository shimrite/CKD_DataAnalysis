import numpy as np


class EDA(object):
    # this section hold the following Data Exploration steps:
    # 1. high level info on the data set
    # 2. features statistics
    # 3. feature correlation
    # 4. data scaling
    # 5. outliers detection
    # 6. Visualization

    def __init__(self, data_input):
        super(EDA, self).__init__()
        self.df_2_explore = data_input
        self.df_mean = []
        self.df_std = []
        self.df_count = []
        self.num_col_list = []
        self.df_outliers = {}
        self.outliers_thresh = 0.01
        self.corr_thresh = 0.7
        self.outliers_col_2b_dropped = []
        self.corr_col_2b_dropped = []

    def info_data(self):
        self.df_2_explore.info()
        self.num_col_list = self.df_2_explore.select_dtypes(include=np.number).columns.tolist()

    def statistics_data(self):
        print(self.df_2_explore.describe())
        self.df_mean = self.df_2_explore.describe().mean()
        self.df_std = self.df_2_explore.describe().std()
        self.df_count = self.df_2_explore.shape[0]   #count()

    def outliers_detection(self):
        # detect outliers (val>mean+2*std)
        # process outliers per feature
        outliers_limit = self.df_count * self.outliers_thresh
        for key in self.num_col_list:
            maxLimit = self.df_mean[key] + 1 * self.df_std[key]
            minLimit = self.df_mean[key] - 1 * self.df_std[key]
            curr_above_std = np.count_nonzero(self.df_2_explore[key] > maxLimit)
            curr_below_std = np.count_nonzero(self.df_2_explore[key] < minLimit)
            self.df_outliers[key] = curr_above_std + curr_below_std
            if self.df_outliers[key] > 0: #outliers_limit:
                self.outliers_col_2b_dropped = self.outliers_col_2b_dropped + [key]
                self.df_2_explore.boxplot(column=key)
        self.df_2_explore.drop(columns=self.outliers_col_2b_dropped)

    def correlated_features_detection(self):
        # Create correlation matrix
        corr_matrix = self.df_2_explore.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than thresh
        self.corr_col_2b_dropped = [column for column in upper.columns if any(upper[column] > self.corr_thresh)] #0.95
        # Drop correlated features
        self.df_2_explore.drop(columns=self.corr_col_2b_dropped)
