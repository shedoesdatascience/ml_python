
# ##  Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 

# For this assessment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. 
# Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32



#1st block of code
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('train.csv',encoding ='latin1')
test_df = pd.read_csv('test.csv')
latlon_df = pd.read_csv('latlons.csv')
addresses_df = pd.read_csv('addresses.csv')

#get list of column names
#print(list(train_df.columns.values))
#print(list(test_df.columns.values))

#get summary stats

#print(new_train_df.describe(include = ['object']))
#which columns are similar
#print(set(train_df.columns).intersection(set(test_df.columns)))

#These columns are not in test set and may be proxy indicators for compliance status
new_train_df = train_df.drop(columns=['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status',
                                     'compliance_detail'])

#remove all rows where compliance is not an issue as these people don't have to pay ticket
new_train_df = new_train_df[pd.notnull(new_train_df['compliance'])]
#remove columns that are similar 
#print(set(new_train_df.columns).intersection(set(test_df.columns)))


#dummify disposition, agency_name(5) 
agency_ind = pd.get_dummies(new_train_df['agency_name'])
disposition_ind = pd.get_dummies(new_train_df['disposition'])
country_ind = pd.get_dummies(new_train_df['country'])
#city_ind = pd.get_dummies(new_train_df['city']) too many weird values to keep

#feature engineering for date values
new_train_df['ticket_issued_date'] = pd.to_datetime(new_train_df['ticket_issued_date'])
new_train_df['hearing_date'] = pd.to_datetime(new_train_df['hearing_date'])



new_train_df['ticket_duration'] = (new_train_df['hearing_date']- new_train_df['ticket_issued_date']).dt.days
new_train_df['ticket_duration'].fillna(-999,inplace = True)
new_train_df['fine_amount'].fillna(-999, inplace = True)



#concatenate dummy variables
new_train_df = pd.concat([new_train_df,agency_ind],axis=1)
new_train_df  = pd.concat([new_train_df,disposition_ind],axis=1)
new_train_df = pd.concat([new_train_df,country_ind],axis=1)
#new_train_df = pd.concat([new_train_df,city_ind],axis=1)
#count number of NA/missing for each column in dataframe
#new_train_df.isnull().sum()

#remove descriptive variables as will create too many dummy vars
new_train2_df = new_train_df.drop(columns=['ticket_id', 'agency_name', 'inspector_name', 'violator_name', 
                                           'violation_street_number','violation_street_name', 'violation_zip_code',
                                           'mailing_address_str_number', 'mailing_address_str_name', 'city',
                                           'state', 'country', 'zip_code', 
                                           'non_us_str_code', 'ticket_issued_date', 'hearing_date', 'violation_code',
                                           'violation_description', 'disposition', 'grafitti_status'])

 
#new_train2_df = new_train_df.dropna(how='all')
#new_train2_df.isnull().sum() #no nas remaining



#define training and validation sets
y_train = new_train2_df['compliance']
X_train = new_train2_df.drop(columns=['compliance'])

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                   random_state = 0)




#fix test set for model
t_agency_ind = pd.get_dummies(test_df['agency_name'])
t_disposition_ind = pd.get_dummies(test_df['disposition'])
#print(t_disposition_ind.head())
t_country_ind = pd.get_dummies(test_df['country'])
#print(t_country_ind.head())
test_df['ticket_issued_date'] = pd.to_datetime(test_df['ticket_issued_date'])
test_df['hearing_date'] = pd.to_datetime(test_df['hearing_date'])



test_df['ticket_duration'] = (test_df['hearing_date']- test_df['ticket_issued_date']).dt.days
test_df['ticket_duration'].fillna(-999,inplace = True)
test_df['fine_amount'].fillna(-999, inplace = True)

test_df = pd.concat([test_df,t_agency_ind],axis=1)
test_df  = pd.concat([test_df,t_disposition_ind],axis=1)
test_df = pd.concat([test_df,t_country_ind],axis=1)


new_test_df = test_df.drop(columns=['ticket_id', 'agency_name', 'inspector_name', 'violator_name', 
                                           'violation_street_number','violation_street_name', 'violation_zip_code',
                                           'mailing_address_str_number', 'mailing_address_str_name', 'city',
                                           'state', 'country', 'zip_code', 
                                           'non_us_str_code', 'ticket_issued_date', 'hearing_date', 'violation_code',
                                           'violation_description', 'disposition', 'grafitti_status'])


X_test = new_test_df
#y_test = new_test_df.drop(columns=['compliance'])

#new_test_df.isnull().sum() #no nas remaining
#new_train_df.isnull().sum() 



def gbm_model(X_train, X_val, X_test,y_train,y_val):
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import GradientBoostingClassifier
    import matplotlib.pyplot as plt

# fit model no training data
    clf = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 5).fit(X_train, y_train)
    

# make predictions for test data
    y_pred = clf.predict(X_val)
    predictions = [round(value) for value in y_pred]
# evaluate predictions
    accuracy = accuracy_score(y_val, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    fpr_lr, tpr_lr, _ = roc_curve(y_val, predictions)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='GBM ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    
    # Your code here
    
    return(accuracy) # Your answer here
gbm_model(X_train, X_val, X_test,y_train, y_val)


#function for randomforest
def randforest_model(X_train, X_val, X_test,y_train,y_val):
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    import matplotlib.pyplot as plt
    
    clf = RandomForestClassifier().fit(X_train, y_train)

# make predictions for test data
    y_pred = clf.predict(X_val)
    predictions = [round(value) for value in y_pred]
# evaluate predictions
    accuracy = accuracy_score(y_val, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    fpr_lr, tpr_lr, _ = roc_curve(y_val, predictions)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='RF ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    
    # Your code here
    
    return(accuracy) # Your answer here

randforest_model(X_train, X_val, X_test,y_train, y_val)

#function for neural network

def nn_model(X_train, X_val, X_test,y_train, y_val):
    from sklearn.neural_network import MLPClassifier
    from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import accuracy_score
    
    nnclf = MLPClassifier(hidden_layer_sizes = 100, solver='lbfgs', alpha=5.0,
                         random_state = 0).fit(X_train, y_train)

# make predictions for test data
    y_pred = nnclf.predict(X_val)
    predictions = [round(value) for value in y_pred]
# evaluate predictions
    accuracy = accuracy_score(y_val, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    fpr_lr, tpr_lr, _ = roc_curve(y_val, predictions)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='NN ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    
    return(accuracy)

nn_model(X_train, X_val, X_test,y_train, y_val)

    
   



#sample code for one-hot encoding
from sklearn.preprocessing import LabelEncoder
dataset = data.values
# split data into X and y
X = dataset[:,0:4]
Y = dataset[:,4]
# encode string class values as integers
encoded_x = None
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	feature = onehot_encoder.fit_transform(feature)
	if encoded_x is None:
		encoded_x = feature
	else:
		encoded_x = numpy.concatenate((encoded_x, feature), axis=1)

