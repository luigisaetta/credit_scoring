import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random as rn
import tensorflow as tf

# to encode categoricals
from sklearn.preprocessing import LabelEncoder

# plt hist of numerical features
def show_dist(df, col, bins, kde=False):
    # plt.title(col)
    sns.histplot(df[col], bins=bins, kde=kde)
    plt.grid()

def plot_hist_numerical(df, num_feat_list, vet_bins):
    plt.figure(figsize=(16,8))
    
    for i, col in enumerate(num_feat_list):
        plt.subplot(2, 4, i + 1)
        # in student_utils
        show_dist(df, col, bins=vet_bins[i])
        plt.grid()
    
    plt.tight_layout()
    
def count_unique_values(df, cat_col_list):
    cat_df = df[cat_col_list]
    val_df = pd.DataFrame({'column': cat_df.columns, 
                       'cardinality': cat_df.nunique() } )
    return val_df

def show_group_stats_viz(df, group):
    print(df.groupby(group).size())
    print(df.groupby(group).size().plot(kind='barh'))
    
# for TF Feature Column API
def normalize_numeric_with_zscore(mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    def norm_func(col):
        col = tf.cast(col, tf.float32)
        
        return (col - mean)/std
    
    return norm_func

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field
    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    tf_numeric_feature = tf.feature_column.numeric_column(col, dtype=tf.dtypes.float32, default_value=default_value, 
                                                          normalizer_fn=normalize_numeric_with_zscore(MEAN, STD))
    
    return tf_numeric_feature

def enable_reproducibility(seed):
    SEED = seed
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is needed for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(SEED)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(SEED)
    tf.random.set_seed(SEED)

def df_to_dataset(df, predictor,  batch_size=32, shuffle=True):
    df = df.copy()
    labels = df.pop(predictor)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    
    # don't shuffle the test set
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    
    ds = ds.batch(batch_size)
    return ds

def split_train_valid(df, predictor_field, batch_size):
    N_TOT = len(df)
    N_TRAIN = int(N_TOT * 0.80)
    N_VALID = N_TOT - N_TRAIN
    
    # shuffle before split
    df = df.sample(frac = 1.)
    df_train = df[:N_TRAIN]
    df_valid = df[N_TRAIN:]
    print('Numero rec. per il training', df_train.shape[0])
    print('Numero rec. per validazione', df_valid.shape[0])
    
    # build the TF dataset
    ds_train = df_to_dataset(df_train, predictor_field,  batch_size=batch_size)
    ds_valid = df_to_dataset(df_valid, predictor_field,  batch_size=batch_size)
    
    return ds_train, ds_valid

#
# functions for categorical encoding
#

# first train label encoder

def train_encoders(df, to_code_list):
    le_list = []

    for col in to_code_list:
        print(f"train for coding: {col} ")

        le = LabelEncoder()
        le.fit(df[col])

        le_list.append(le)

    print()

    return le_list


# then use it
def apply_encoders(df, le_list, to_code_list):

    for i, col in enumerate(to_code_list):
        print(f"Coding: {col} ")

        le = le_list[i]

        df[col] = le.transform(df[col])

    # special treatment for windspeed
    # windpeed actually is integer badly rounded !!
    # print('Coding: windspeed')
    # df['windspeed'] = np.round(df['windspeed'].values).astype(int)

    return df
