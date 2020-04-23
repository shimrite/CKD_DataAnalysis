import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('C:\\Users\\shimr\\Documents\\work\\tests\\testElminda\\CKD.csv', index_col = 0)
print(df.shape)
df.info()
df.drop_duplicates(inplace=True)
print(df.shape)
print(df.columns)
# extract labels col
y_data = df['classification']
x_data = df.drop(columns = ['classification'])
# split data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33, random_state=48)
print(x_train.shape)
# -- handle train data --
# imputation
x_train_val = x_train.dropna(axis=1, thresh=x_train.shape[0]*0.75)
print(x_train_val.shape)
train_val_columns = x_train_val.columns
# clean data
# impute categorical val
cat_values = {} # create categorical common values dictionary - per att its most common value (top)
cat_col_list = x_train_val.select_dtypes(include=np.object).columns.tolist()
for key in cat_col_list:
    x_train_val[key] = x_train_val[key].str.strip()
    x_train_val[key] = x_train_val[key].str.strip('?')
    x_train_val[key] = pd.to_numeric(x_train_val[key], errors='ignore') # e.g. pcv
    if x_train_val[key].dtype == object:
        cat_values[key] = x_train_val[key].describe().top
cat_col_list = x_train_val.select_dtypes(include=np.object).columns.tolist() # after removal of numerical columns
x_train_val.fillna(value=cat_values, inplace=True)
# impute numerical val (mean/median/corr)
num_desc = x_train_val.describe() # note, this is executed only on numerical data
mean_num_values = num_desc.loc['mean']
x_train_val.fillna(value=mean_num_values, inplace=True)

# scale train data
x_train_val_scale = x_train_val
mms = MinMaxScaler() # fit numerical data
num_col_list = x_train_val.select_dtypes(include=np.number).columns.tolist()
x_train_val_scale[num_col_list] = mms.fit_transform(x_train_val_scale[num_col_list])
cat2num_map = {}
for key in cat_col_list: # create categorical values dictionary - per att its optional values and their numerical replacement (top)
    cat2num_map[key] = {x_train_val[key].value_counts().keys()[0]: 0, x_train_val[key].value_counts().keys()[1]: 1}
x_train_val_scale.replace(cat2num_map, inplace=True) # scale categorical data (TBD - onehot!)

x_train_val_scale.info()

# -- handle test data -- get only valid columns, fill NA, scale/fit
x_test_val = x_test[train_val_columns]
# clean data
cat_col_list_test = x_test_val.select_dtypes(include=np.object).columns.tolist()
for key in cat_col_list_test:
    x_test_val[key] = x_test_val[key].str.strip() # note, str can be triggered only per colums (cant be performed on the all data frame)
    x_test_val[key] = x_test_val[key].str.strip('?')
    x_test_val[key] = pd.to_numeric(x_test_val[key], errors='ignore')  # e.g. pcv
cat_col_list_test = x_test_val.select_dtypes(include=np.object).columns.tolist() # after removal of numeric
print(cat_col_list_test)
print(cat_col_list) # check the same
# fill NA
x_test_val.info()
x_test_val.fillna(value=mean_num_values, inplace=True)
x_test_val.fillna(value=cat_values, inplace=True)
# scale
x_test_val_scale = x_test_val
x_test_val_scale.replace(cat2num_map, inplace=True)
x_test_val_scale[num_col_list] = mms.fit_transform(x_test_val_scale[num_col_list])
x_test_val_scale.info()


# scale labels to 0/1
y_cat2num_map={}
# for key in y_train.keys():
y_train = y_train.str.strip()
y_test = y_test.str.strip()
y_cat2num_map = {y_train.value_counts().keys()[0]: 0, y_train.value_counts().keys()[1]: 1}
y_train.replace(y_cat2num_map, inplace=True)
y_test.replace(y_cat2num_map, inplace=True)


# feature selection
# method 1 - K best by chi^2 statistic
test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(x_train_val_scale, y_train)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(x_train_val_scale)
# summarize selected features
print(features[0:15,:]) # print data of the K best features
print(np.argsort(fit.scores_)[-10:]) # best K features indexes (10)
print(x_train_val_scale.values[:15,np.argsort(fit.scores_)[-10:]]) # print data of the K best features

a=[1,2,3]

