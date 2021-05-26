
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import scipy.stats as stats
import itertools
import pickle

### sklearn
from sklearn.utils import shuffle
# This is for the Learning curve, hyperparam tuning and KFold CV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedShuffleSplit, learning_curve, ShuffleSplit, train_test_split
from sklearn.metrics import make_scorer
# Modelling
from sklearn.ensemble import RandomForestClassifier
# Classification metrics
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, fbeta_score

################################## Data Cleansing 

def find_bad_columns_function(dataframe):
    '''
    Args: dataframe for which there maybe bad columns
    
    Logic: Find the columns that have nulls, blanks, quasi constant/constant values defined by less than 1% variance or duplicates
    
    Returns: lists containing names of those features that have nulls, blanks, constant values throughout or 
    those that are duplicates of other features
    
    '''
    # I could have passed in just the features instead of df. But i chose to keep the target column
    # because you never know if there are null values there !
    
    ###### Finding Null Values
    null_col_list = dataframe.columns[dataframe.isna().any()].tolist()
    
    print('Identified {} features with atleast one null'.format(
        len(null_col_list)))

    ###### Finding Blank Spaces in the object column
    # Non-obvious nulls such as blanks: The line items where there are spaces 
    blank_space_col_list = []
    object_columns = dataframe.select_dtypes(include=['object']).columns

    for col in object_columns:

        if sum(dataframe[col]==' '):
            blank_space_col_list.append(col)

    print('Identified {} features with atleast one blank space'.format(
        len(blank_space_col_list)))
    
    ####### Finding Quasi Constant/Constant Value in numerical columns
    # removing the variables that have more than 99% of their values as the same 
    # ie their standard deviation is less than 1 %
    
    numeric_df = dataframe._get_numeric_data()

    constant_numeric_col_list = [col for col in numeric_df.columns if numeric_df[col].std()<0.01]

    print('Identified {} numeric features that have quasi-constant values'.format(
        len(constant_numeric_col_list)))
    
    # Didnt use the following code snippet for the above because if you have closely varying float values
    # then the below wont pick it up
    
    ###### Finding Quasi Constant/Constant non_numeric value
    constant_non_numeric_col_list = []
    
    # Find the columns that are not in numeric_df
    non_numeric_col_set = set(dataframe.columns) - set(numeric_df.columns)   

    for col in non_numeric_col_set:
        categorical_mode_value = (dataframe[col].mode().values)[0]
        fractional_presence = sum(dataframe[col]==categorical_mode_value)/len(dataframe) 
    
        if fractional_presence > 0.99:
            constant_non_numeric_col_list.append(col)
            
    print('Identified {} non-numeric features that have quasi-constant values'.format(
        len(constant_non_numeric_col_list)))
    
    ####### Finding Duplicate columns
    
    # Aliter: Use the pandas .drop_duplicates to transpose first, keep the non_duplicates then transpose back
    # Then subtract the column names to find out what is missing from the latter term
    # Inspired by https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
    duplicate_col_list = set(dataframe.columns) - set((dataframe.T.drop_duplicates().T).columns)
    
    # The above assumes that the column NAMES are not duplicated (which is fairly easy to check outside of this function)

    print('Identified {} features that are duplicates of other features'.format(
        len(duplicate_col_list)))
    
    return null_col_list, blank_space_col_list, constant_numeric_col_list, constant_non_numeric_col_list, duplicate_col_list

# null_col_list, blank_space_col_list, constant_numeric_col_list, \
# constant_non_numeric_col_list, duplicate_col_list = find_bad_columns_function(df)

def prefix_suffix_removal_function(dataframe, prefix_col_dict={}, suffix_col_dict={}):
    '''
    Args: dataframe containing columns containing prefixes and/or suffixes which are identified
    by the prefix_col_dict and suffix_col_dict. These dictionaries need to contain the column name
    as the key and the prefix or suffix as the string value
    
    Logic: The columns are first converted to string and prefixes/suffixes are iteratively removed
    Returns: the cleaned dataframe
    
    '''
    # dictionary is used instead of a list because a specific suffix with a column 
    # Removing suffixes, example $100 -> 100
    for col, prefix in prefix_col_dict.items():
        dataframe[col] = dataframe[col].str.replace(prefix,"")
        print('Prefix {} removed for column {}'.format(prefix, col))
        
    # Removing suffixes, example 100% -> 100
    for col, suffix in suffix_col_dict.items():
        dataframe[col] = dataframe[col].str.replace(suffix,"")
        print('Suffix {} removed for column {}'.format(suffix, col))
        
    return dataframe


# Note that even though the same snippet of code can remove prefix and a suffix, I have chosen to separate it out
# Because there can be instances where the column contains both the prefix and a suffix

# This is how you would call it
# prefix_col_dict = {'dummy_col':'$'}
# suffix_col_dict = {'dummy_col2':'%'}
# prefix_suffix_removal_function(temp_df, prefix_col_dict, suffix_col_dict)

def time_feature_engineering_function(dataframe, time_column):
    
    """
    Args: dataframe and the timeseries column called time_column
    Assumption: time_column already has the hour day month year information
    Logic: generate 8 possible combinations of the time_column column 
    Returns: The original dataframe with the new feature engineered columns + names of those columns
    
    """
    
    # Then use the above column to split up into respective time components 
    # in which there might be seasonality
    dataframe['hour'] = dataframe[time_column].dt.hour
    
    dataframe['dayofweek'] = dataframe[time_column].dt.dayofweek
    dataframe['dayofmonth'] = dataframe[time_column].dt.day
    dataframe['dayofyear'] = dataframe[time_column].dt.dayofyear
    
    dataframe['weekofyear'] = dataframe[time_column].dt.weekofyear
    dataframe['quarter'] = dataframe[time_column].dt.quarter
    dataframe['month'] = dataframe[time_column].dt.month
    dataframe['year'] = dataframe[time_column].dt.year
    
    new_col_name_list = ['hour','dayofweek','dayofmonth','dayofyear','weekofyear','quarter','month','year']
    
    return dataframe, new_col_name_list

########################################################### Data Conversion

# this is part of the reduce_mem_usage_function()
def null_imputation_function(dataframe, col, NA_dict, imputed_val):
    '''
    Args: dataframe whose col has nulls that has to be imputed with imputed_val
    NA_dict records the above columns
    
    Returns: the imputed dataframe column, dictionary with {null_col_name: imputed_val} and indices of all rows with nulls
    
    '''
    
    # Store the existence of nulls in this column as a flag that can be utilized again after the data conversion 
    # because Integer datatype does not support NA, therefore, NA needs to be filled
    null_existence_col_flag = np.isfinite(dataframe[col]).all()
    # Note that np.isfinite is equally applicable for both np.nan and infinite numbers
    # It will return False if there is atleast 1 null or infinite value in the column
    
    null_indices_list = []
    
    #Check if every item in a column is finite, otherwise fill nulls with the (minimum value - 1)
    if not null_existence_col_flag: 
        
        # Store the null indices in a list
        null_indices_list = dataframe[dataframe[col].isnull()].index.tolist()
        
        dataframe[col].fillna(imputed_val, inplace=True)  

        # Record the imputed value so that we can bring it back outside of the function if needed
        NA_dict[col] = imputed_val
        
    return dataframe[col], null_existence_col_flag, NA_dict, null_indices_list


def float_to_int_conversion_function(dataframe, col, mn, mx):
    '''
    Args: dataframe containing the col which has been proven previously that it can be converted to an int
    mn, mx are the minimum and maximum values of the column 
    
    Note that the dataframe should not have any nulls 
    Logic: The code downcasts the float64 into uint8,16,32,64 if its a strictly positive integer or 
    int8,16,32, or 64 if its a signed integer
    
    Returns: dataframe column with the changed datatype
    '''
    
    #     #if the minimum is positive, then the numbers need unsigned integers
    if mn >= 0:
        if mx < 255:
            dataframe.loc[:,col] = dataframe[col].astype(np.uint8)
        elif mx < 65535:
            dataframe.loc[:,col] = dataframe[col].astype(np.uint16)
        elif mx < 4294967295:
            dataframe.loc[:,col] = dataframe[col].astype(np.uint32)
        else:
            dataframe.loc[:,col] = dataframe[col].astype(np.uint64)
    else:
        #allocate signed integers
        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
            dataframe.loc[:,col] = dataframe[col].astype(np.int8)
        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
            dataframe.loc[:,col] = dataframe[col].astype(np.int16)
        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
            dataframe.loc[:,col] = dataframe[col].astype(np.int32)
        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
            dataframe.loc[:,col] = dataframe[col].astype(np.int64)
    
    return dataframe[col]

# Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage_function(dataframe, print_ongoing_column_change = True, percentile_threshold = 0.01):
    
    '''
    Args: The dataframe to be compressed; a boolean variable called print_ongoing_column_change, that indicates 
    the ongoing variable change gets printed or not; 
    and percentile_threshold to make criterion to convert floats to ints more stringent
    
    Logic: The column numbers are hardcoded as 8, 16, 32 or 64 bit depending on the range of cell vals
    columns with some infinite values are hardcoded with nulls with min - 1
    
    Returns: the compressed dataframe and a dictionary NA_dict containing the col names as keys and their imputed values as values
    
    '''
     
    #Calculate the memory usage in MB. This scaling factor is more precise than dividing by 1e6
    start_mem_usg = dataframe.memory_usage().sum() / 1024**2 
    print("Memory usage of dataframe is {} MB".format(round(start_mem_usg,2)))

    # This dictionary is really just for audit purposes 
    # to keep track of which columns had nulls/infinites which got replaced with min() -1 and then replaced back
    NA_dict = Counter()
       
    for col in dataframe.columns:
        if dataframe[col].dtype != object:  # Exclude strings
            
            # make variables for Int, max and min
            IsInt = False
            mx = dataframe[col].max()
            mn = dataframe[col].min()

            # You can change how strict the comparison is using the percentile_threshold

            # If we are forcing conversion of floats to int, we want 99% of our numbers to be above 10 
            # (which is really 10*0.99 ie 10x the maximum value of the error due to rounding off the decimal)
            IsInt = np.abs(dataframe[col].quantile(q=percentile_threshold)) > 10

            # Make sure to do the above before imputation of nulls 
            # (otherwise your statistical measures will be skewed)
            
            # Make Integer/unsigned Integer datatypes for floats that are large enough compared to the decimal
            if IsInt:   

                dataframe.loc[:,col], null_existence_col_flag, \
                NA_dict, null_indices_list = null_imputation_function(dataframe, col, NA_dict,imputed_val = mn-1)

                dataframe.loc[:,col] = float_to_int_conversion_function(dataframe, col, mn, mx)

                # Bring back the nulls in those indices we stored before for the subset of the columns for which nulls originally existed
                if not null_existence_col_flag:
                    print(col, len(null_indices_list))
                    dataframe.loc[null_indices_list, col] = np.nan

            # Make float datatypes 32 bit only when the underlying numbers are not ints
            else:
                dataframe.loc[:,col] = dataframe[col].astype(np.float32)                
            
            if print_ongoing_column_change:
                
                # Print new column type
                print("Column {} dtype after is {} ".format(col, dataframe[col].dtype))
                print("******************************")
    
    # Print final result
    print("___Memory usage after Data size reduction:___")
    
    mem_usg = dataframe.memory_usage().sum() / 1024**2 
    print("Memory usage is {} MB ".format(round(mem_usg,2)))
    
    print("Final data size is {} percent of initial size".format(round((100*mem_usg/start_mem_usg),2)))
    return dataframe, NA_dict

# df, NA_dict = reduce_mem_usage_function(df)






    
########################################################### ML Modelling    

def roc_curve_function(fpr, tpr):
    
    '''
    Args: The False Positive and True Positive rates
    Plots: The ROC plot 
    
    '''    
    
    plt.figure(figsize=(10,6))
    
    plt.title('ROC Curve', fontsize=16)
    
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, 'b-', linewidth=2)
    
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    

def plot_confusion_matrix_function(cm, classes,normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    # This is from sklearn's example page 
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()
    
def custom_classification_metrics_function(X_test, y_test, labels, classifier, data_type='Test', output_metrics=False):
    
    '''
    Args: The features and the target column; the labels for the target column; sklearn classifier object
    that has been fitted with the training data outside of the function; whether we are running it for Train or Test; and finally whether we want to generate the output metrics as a dictionary
    
    Logic: Calculates and displays the Accuracy, Classification report, ROC Curve, Confusion Matrix
    
    Returns: Dictionary containing relevant metrics for classification (only if the output_metrics flag is set)
    
    '''
    
    ### Plain old accuracy
    test_score = classifier.score(X_test, y_test)
    
    # change the floating point representation   
    test_accuracy = float("{0:.2f}".format(test_score.mean()))
    print("Has a {} accuracy of {} % ".format(data_type, test_accuracy * 100))

    ### Classification report
    test_pred = classifier.predict(X_test)

    print(classification_report(y_test, test_pred, target_names=labels))
    
    ### ROC AUC score
    # Only store the 2nd column which corresponds to the probability for the positive class
    y_scores = classifier.predict_proba(X_test)[:, 1]

    # change the floating point representation                     
    roc_auc_val = float("{0:.2f}".format(roc_auc_score(y_test, y_scores)))
    print('The ROC AUC Score on {} set is {}'.format(data_type, roc_auc_val))

    ### ROC AUC Curve                      
    fpr, tpr, threshold = roc_curve(y_test, test_pred)   
    roc_curve_function(fpr, tpr)

    ### Confusion Matrix
    confusion_matrix_test_object = confusion_matrix(y_test, test_pred)

    fig = plt.figure(figsize=(8,4))
    plot_confusion_matrix_function(confusion_matrix_test_object, 
                                   labels, title= data_type + ' ' + "Confusion Matrix", cmap=plt.cm.Oranges)
       
    ### Output metrics (optional)
    if output_metrics:
        
        # Initialize a dictionary to store the 8 metrics we are interested in
        metrics_dict = Counter()

        # Metrics from the confusion matrix
        metrics_dict['TN'] = confusion_matrix_test_object[0][0]
        metrics_dict['TP'] = confusion_matrix_test_object[1][1]
        metrics_dict['FN'] = confusion_matrix_test_object[1][0]
        metrics_dict['FP'] = confusion_matrix_test_object[0][1]

        # Metrics from the classification report
        # We are only interested in the minority class which is 1
        metrics_dict['Pos_Precision'] = classification_report(y_test, test_pred, output_dict=True)['1']['precision']
        metrics_dict['Pos_Recall'] = classification_report(y_test, test_pred, output_dict=True)['1']['recall']
        metrics_dict['Pos_F1'] = classification_report(y_test, test_pred, output_dict=True)['1']['f1-score']

        # Metric from the ROC AUC curve
        metrics_dict['ROC AUC'] = roc_auc_val

        return metrics_dict
    
    
    
def precision_at_recall_threshold_function(y_test, predicted_proba, recall_threshold=0.85):
    '''
    Args: The true labels, predicted probabilities and the threshold for which recall needs to be computed
    
    Logic: Using the precision recall curve method from sklearn, the precision, recall and threshold are calculated
    
    Returns: "What was the precision when recall was just above the recall_threshold?" 
    
    '''
    # from sklearn.metrics import precision_recall_curve
    
    precision, recall, _ = precision_recall_curve(y_test, predicted_proba)
    
    return precision[recall>recall_threshold][-1]





###########################################################################

def numerical_distribution_function(dataframe, showOutliers = True, bins = 10):
    '''
    Args: dataframe containing the numerical columns for which the plots need to be displayed 
    Plots: The histogram, boxplots and the quantile plot allowing you to visually identify the outliers and the need to transform
    
    '''
    # The good part about this function is that you dont need to worry about nulls in the data
    
    for col in dataframe.columns:

        plt.figure(figsize=(16,8))
        
        # Histogram
        plt.subplot(1,3,1)
        fig = dataframe[col].hist(bins=bins)
        fig.set_title('Histogram of ' + col)
        fig.set_ylabel('Counts')
        fig.set_xlabel(col)
        
        # Boxplot
        plt.subplot(1,3,2)
        fig = sns.boxplot(dataframe[col], orient='v', width =0.3)
        fig.set_title('Boxplot of ' + col)
        fig.set_ylabel(col)
        
        # Quantile plot
        plt.subplot(1,3,3)
        stats.probplot(dataframe[col],dist="norm",plot=plt)

        plt.show
# numerical_distribution_function(numeric_df)

def merge_feature_target_train_test_function(X_train, X_test, y_train, y_test):
    '''
    Args: Feature and target for train and test set passed as dataframes
    Returns: combined_dataframe containing all the data
    
    '''
    temp_X_train = X_train.reset_index(drop=True)
    temp_y_train = y_train.reset_index(drop=True)

    temp_X_test = X_test.reset_index(drop=True)
    temp_y_test = y_test.reset_index(drop=True)

    temp_xy_train = pd.concat( [temp_X_train, temp_y_train], axis=1) 
    temp_xy_test = pd.concat( [temp_X_test, temp_y_test], axis=1) 

    combined_dataframe = pd.concat( [temp_xy_train, temp_xy_test], axis=0)
    
    return combined_dataframe
# null_imputed_df = merge_feature_target_train_test_function(X_train, X_test, y_train, y_test)


def data_export_function(X_train, X_test, y_train, y_test, 
                         intermediate_data_path, output_df_file_name, export_date):
    '''
    
    Logic: Combine the X_train with X_test row-wise. Similarly combine y_train with y_test

    Returns: dataframe which can be used directly for modelling purposes
    '''
    
    temp_X_train = X_train.reset_index(drop=True)
    temp_y_train = y_train.reset_index(drop=True)

    temp_X_test = X_test.reset_index(drop=True)
    temp_y_test = y_test.reset_index(drop=True)

    temp_xy_train = pd.concat( [temp_X_train, temp_y_train], axis=1) 
    temp_xy_test = pd.concat( [temp_X_test, temp_y_test], axis=1) 

    temp_xy = pd.concat( [temp_xy_train, temp_xy_test], axis=0) 
    
    #export the data to be used
    temp_xy.to_csv(intermediate_data_path + output_df_file_name + export_date + '.csv', index=False)
    
# 11th of April 2020: created a wrapper around the usual model fitting and displaying classification metrics
def test_rf_performance_function(X_train, y_train, X_test, y_test, labels, recall_threshold=0.85):
    
    '''
    Args: The train and test data; labels to allow a human readable graph
    
    Logic: Trains a random forest classifier with default params and outputs the classification report & confusion matrix
    
    Displays: The confusion matrix and classification report
    
    Returns: the metrics_dict containing the necessary binary metrics
    '''

    from sklearn.ensemble import RandomForestClassifier

    # Initialize a classifier object with default params
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Display the classification report on the Test data
    metrics_dict = custom_classification_metrics_function(
        X_test, y_test, labels, classifier, data_type='Test', output_metrics = True) 
         
    predicted_proba = classifier.predict_proba(X_test)[:,1]
    threshold_precision = precision_at_recall_threshold_function(y_test, predicted_proba, recall_threshold=0.85)
    
    metrics_dict['Precision_at_Recall'] = threshold_precision
    
    return metrics_dict
# test_rf_performance_function(X_train, y_train, X_test, y_test, labels, output_metrics = True)


def resample_and_test_performance_function(ResampleMethod, X_train, y_train, X_test, y_test, labels):
    '''
    Args: imblearn method that has the fit_sample method to generate samples;
    the train, test data and the labels
    
    Logic: The imblearn method is called to generate the synthetic data
    
    Returns: the metrics dictionary which is actually generated by a separate function 
    
    '''
    # Generate the resampled data using one of the imblearn methods
    X_resampled_train, y_resampled_train = ResampleMethod.fit_sample(X_train, y_train)
    
    print('The size of the resampled train sets are as follows:')
    print(X_resampled_train.shape,sum(y_resampled_train))
    
    # call the function to generate the plots and return the metrics
    metrics_dict  = test_rf_performance_function(
                            X_resampled_train, y_resampled_train, X_test, y_test, labels)
    
    return X_resampled_train, y_resampled_train, metrics_dict

# _,__, sampling_strategy_metrics_df.loc['Undersampling'] = resample_and_test_function(
#                                       RandomUnderSampler(), X_train, y_train, X_test, y_test, labels)


# This function was inspired by the notebook from Kaggle
# https://www.kaggle.com/niteshx2/beginner-explained-lgb-2-leaves-augment
def augment_data_function(X_train, y_train, positive_upsampling_ratio=4, negative_upsampling_ratio=2): 
    
    '''
    Args:
    X_train - feature dataframe
    y_train - target series
    positive_upsampling_ratio - how much to upsample the positive class by
    negative_upsampling_ratio - how much to upsample the negative class by
    
    Logic: The two separate upsampling ratios allow you to independently control how much each class gets sampled
    While this was originally created to augment both classes simultaneously, 
    it can be used to deliberately upsample one class but not the other    
    
    Returns: augmented feature and target columns
    
    '''    
    # Always set the seed ! Otherwise you wont be able to replicate the results
    np.random.seed(42)
    
    # Convert the input dataframes into arrays
    x, y = X_train.values, y_train.values
    
    # create empty arrays for the positive and negative rows in the input data
    xs, xn = [], []
    
    # we need the -1 in the for loop range() to make sure that the overall upsampling ratio is what we want
    # This code segment augments the positive class
    for i in range(positive_upsampling_ratio-1):
        
        # Create a mask for the positive class and augment those indices
        mask = y>0
        temp_array = x[mask].copy()
        # create an array of indices from the number of rows 
        ids = np.arange(temp_array.shape[0])
        
        # This is the core of the augmentation procedure. The rows are shuffled within a given column
        for c in range(temp_array.shape[1]):
            
            # this shuffles the indices in place
            np.random.shuffle(ids)
            temp_array[:,c] = temp_array[ids][:,c]
            
        xs.append(temp_array)

    # This code segment augments the negative class
    for i in range(negative_upsampling_ratio-1):
        
        # Create a mask for the negative class and augment those indices
        mask = y==0
        temp_array = x[mask].copy()
        ids = np.arange(temp_array.shape[0])
        
# This is the core of the augmentation procedure. The rows are shuffled within a given column
        for c in range(temp_array.shape[1]):
            np.random.shuffle(ids)
            temp_array[:,c] = temp_array[ids][:,c]
        
        # Append these back to the array that stores the negative labelled features
        xn.append(temp_array)
        
    
    xs = np.vstack(xs)
    xn = np.vstack(xn)
    
    # create an array of 1s and 0s with the same number of rows as the corresponding synthetically created feature matrices
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    
    # merge it back with the original data by stacking it one below the other
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    
    # convert the arrays back into dataframes
    X_resampled_df = pd.DataFrame(data=x, columns=X_train.columns)
    y_resampled = pd.Series(y)
    
    # Shuffle the merged dataframe so that 1s and 0s are not right below each other
    X_resampled_df, y_resampled = shuffle(X_resampled_df, y_resampled)

    return X_resampled_df, y_resampled

### X_train_resampled, y_train_resampled = augment_data_function(X_train, y_train,  
#                                                    positive_upsampling_ratio=4, negative_upsampling_ratio=2)

### Typical Usage: within the stratified K Fold when you are augmenting the train portion and predicting on the other fold

# Limitations:
#     Only applicable for binary classes. If you want to use it for multi-class classification, then you would have to 
#     generalize the code starting with how the input is received


# This is adapted from https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection
def plot_feature_importances(dataframe, threshold = 0.95):
    """
    Args: dataframe (whose columns must be feature and importance) and 
    threshold (with a default value of 0.95)
    
    Logic:  Plots upto 15 most important features and the cumulative importance of features.
    Also prints the number of features needed to reach threshold cumulative importance.
    
    Returns: Dataframe ordered by feature importances with 2 additional columns - 
        a normalized column and a cumulative importance column
    
    """
    
    # Sort features in descending order of importance inplace
    dataframe = dataframe.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    dataframe['importance_normalized'] = dataframe['importance'] / dataframe['importance'].sum()
    dataframe['cumulative_importance'] = np.cumsum(dataframe['importance_normalized'])

    ##### Horizontal bar chart of feature importances
    
    # Initialize plotting params
    plt.rcParams['font.size'] = 18
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # can be parameterized in the function argument if needed
    no_of_features_to_plot = 15     
    
    # This is needed due to a quirk in barh() where the graph gets built from bottom to top
    # Reverse the dataframe row numbers so that the least important feature is the first element
    yaxis_ticks = list(reversed(list(dataframe.index[:no_of_features_to_plot])))
    
    # aliter use (list(dataframe.index[:no_of_features_to_plot]).reverse()
    
    # Bar plot laid out horizontally
    ax.barh(yaxis_ticks, dataframe['importance_normalized'].head(no_of_features_to_plot), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(yaxis_ticks)
    ax.set_yticklabels(dataframe['feature'].head(no_of_features_to_plot))
    
    
    # Plot labelling
    plt.xlabel('Normalized Importance')
#     plt.ylabel('Feature Names')
    plt.title('Feature Importances')
    plt.show()
    
    ###### Cumulative importance plot
    plt.figure(figsize = (8, 6))
    
    plt.plot(list(range(len(dataframe))), dataframe['cumulative_importance'], 'r-')
    
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance'); 
    plt.title('Cumulative Feature Importance');
    plt.show();
    
    ##### Printing the number of features needed to reach the threshold
    # You are adding +1 because the dataframe indexing starts at 0.
    no_of_important_col = np.min(np.where(dataframe['cumulative_importance'] > threshold)) + 1
    
    print('{} features required for {} of cumulative importance'.format(
                                        no_of_important_col, threshold))
    
    return dataframe, no_of_important_col

# norm_feature_importances_df,no_of_important_col = plot_feature_importances(feature_importances_df)


# Included on the 12th of April
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    # call the sklearn learning_curve function 
    # that returns an array of train sizes and associated train/test scores
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    
    ############### Plotting 
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    
    plt.grid()

    # Create a fuzzy margin around the main trend line with outerbounds defined by +/- std deviation
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    return plt

def metrics_store_function(optimised_model, cv_fold_X_val, cv_fold_y_val):
    
    '''
    Args: The model with the best combo of hyperparams, X & Y validation data
    Logic: After generating the predictions using the passed model, 
    calls the builtin metrics for scoring on the above predictions
    Returns: The Accuracy, Precision, Recall, F1 Score, AUC ROC
    
    '''
    
    from collections import Counter
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    classification_metrics_dict = Counter()
    # Predict on the validation folds that have NOT been upsampled
    val_pred = optimised_model.predict(cv_fold_X_val)
    
    classification_metrics_dict['accuracy'] = optimised_model.score(cv_fold_X_val, cv_fold_y_val)
    classification_metrics_dict['precision'] = precision_score(cv_fold_y_val, val_pred)
    classification_metrics_dict['recall'] = recall_score(cv_fold_y_val, val_pred)
    classification_metrics_dict['f1'] = f1_score(cv_fold_y_val, val_pred)
    classification_metrics_dict['roc_auc'] = roc_auc_score(cv_fold_y_val, val_pred)

    return classification_metrics_dict

def hyper_param_search_function(X_train, y_train, hyper_param_space, 
                                classifier_type = RandomForestClassifier, n_iter = 30,
                                 search_method = 'random_search', resampling=False, **kwargs):
    '''
    Args: training data + hyperparam search space according to the sklearn classifier being passed;
    **kwargs is for the resampling params like upsampling ratios for my custom data augmentation function
    
    The search_method specifies whether its a RandomSearchCV or a GridSearchCV being passed inside
    
    '''
    # This is to store the best_params and the roc from each fold 
    hyperparam_results = Counter()

    cv_fold_X_train, cv_fold_X_val, cv_fold_y_train, cv_fold_y_val = train_test_split(X_train, y_train, 
                                                        test_size=0.2, random_state=42, stratify=y_train)

    # Instantiate a classifier (eg:RandomForest) object along with the hyperparam range
    if search_method == 'random_search':
        search_object = RandomizedSearchCV(classifier_type(), 
                                  hyper_param_space, cv = 5, n_iter = n_iter,
                                  verbose=4, random_state=42, n_jobs = -1)

    elif search_method == 'grid_search':
        search_object = GridSearchCV(classifier_type(), 
                                  hyper_param_space, cv = 5, 
                                  verbose=4, n_jobs = -1)

    # I have deliberately used the if-else statement because 
    # there are slight syntactical differences between the parameter inputs for Random vs Grid Search

    if resampling:
        X_train_resampled, y_train_resampled = augment_data_function(cv_fold_X_train, cv_fold_y_train,  
                                                   positive_upsampling_ratio=4, negative_upsampling_ratio=2)
        
        # Actually fitting the model on the subset of the training data
        model = search_object.fit(X_train_resampled, y_train_resampled)
    else:
        # Actually fitting the model on the subset of the training data
        model = search_object.fit(cv_fold_X_train, cv_fold_y_train)

    # Potential improvement to the above would be to have another if statement inside where you are either selecting the
    # custom function or any of the methods passed. This would also require a change in the argument
    # We really need just 2 sets of data: the best hyperparam combo and validation performance estimate
    hyperparam_results['best_params'] = model.best_params_
    
    # Get the model with the best combination of hyperparams, use it to predict on validation fold
    optimised_model = model.best_estimator_

    # .... and store all the relevant classification metrics
    classification_metrics_dict = metrics_store_function(optimised_model, cv_fold_X_val, cv_fold_y_val)
  
    hyperparam_results['classification_metrics'] = classification_metrics_dict

    ######### Pretty print
    print('---' * 45)
    print('')

    print("Accuracy: {}".format(round(classification_metrics_dict['accuracy'],4)))
    print("Precision: {}".format(round(classification_metrics_dict['precision']),4))
    print("Recall: {}".format(round(classification_metrics_dict['recall'],4)))
    print("F1: {}".format(round(classification_metrics_dict['f1'],4)))
    print("AUC ROC: {}".format(round(classification_metrics_dict['roc_auc'],4)))

    print('---' * 45)
    
    return hyperparam_results

def tune_grid_search_function(cv_fold_X_train, cv_fold_y_train, cv_fold_X_val, cv_fold_y_val,
hyper_param_space, classifier_type = RandomForestClassifier, **kwargs):
    
    '''
    Args: the train and test folds, a dictionary of the hyperparams for the uninstantiated classifier model
    the classifier type and any key word args 
    including those for custom scoring functions passed to the GridSearchCV
    
    Logic: GridSearchCV is run for the above combo (stratification is on by default)
    
    The metrics are calculated with the optimised model on the validation fold 
    
    Returns: hyperparam_results contains the best params for the classifier model; relevant metrics
    
    '''
    from collections import Counter
    # This is to store the best_params and metrics from each fold 
    hyperparam_results = Counter()
    
    if kwargs['is_custom_scorer']:
        
        # get the custom_scoring_function object from the kwargs dictionary
        custom_scoring_function = kwargs['custom_scoring_function']
        
        # ... and pass it onto sklearn's make_scorer
        from sklearn.metrics import make_scorer
        custom_scorer = make_scorer(custom_scoring_function, greater_is_better=True, needs_proba=True)

        search_object = GridSearchCV(classifier_type(), 
                                      hyper_param_space, scoring=custom_scorer, cv = 3, 
                                      verbose=10, n_jobs = 1)

    else:
        # Notice how the GridSearchCV doesnt have the scoring param
        search_object = GridSearchCV(classifier_type(), 
                                  hyper_param_space, cv = 3, 
                                  verbose=10, n_jobs = 1)
    
    # Actually fitting the gridsearch object on the subset of the training data
    model = search_object.fit(cv_fold_X_train, cv_fold_y_train)
    
    # Allows the later plotting of the grid search improvements over time
    internal_cv_score_list = [x for x in model.cv_results_['mean_test_score']]
    
    hyperparam_results['internal_grid_search_scores'] = internal_cv_score_list
    
    # We really need just 2 sets of data: the best hyperparam combo and validation performance estimate
    hyperparam_results['best_params'] = model.best_params_
       
    
    # Get the model with the best combination of hyperparams, use it to predict on validation fold
    optimised_model = model.best_estimator_
    

    # .... and store all the relevant classification metrics
    classification_metrics_dict = metrics_store_function(optimised_model, cv_fold_X_val, cv_fold_y_val)
  
    classification_metrics_dict['f2'] = fbeta_score(cv_fold_y_val, optimised_model.predict(
                            cv_fold_X_val), average='weighted', beta=2)
    
    if kwargs['custom_scoring_function'] == precision_at_recall_threshold_function:
        
        recall_threshold = kwargs['recall_threshold']
        
        classification_metrics_dict['precision_at_recall'+ str(recall_threshold)] = precision_at_recall_threshold_function(
            cv_fold_y_val, optimised_model.predict_proba(cv_fold_X_val)[:,1], 
            recall_threshold=recall_threshold)
    
    hyperparam_results['classification_metrics'] = classification_metrics_dict
    
    return hyperparam_results

#hyperparam_results = tune_grid_search_function(
# X_train_resampled, y_train_resampled, cv_fold_X_val, cv_fold_y_val,
#  rf_params, classifier_type = classifier_type, is_custom_scorer = True, 
# custom_scoring_function =  precision_at_recall_threshold_function, recall_threshold=0.85)

# If you are not using the custom_scoring_function, 
# then set is_custom_scorer = False, custom_scoring_function = False