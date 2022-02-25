import valohai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

valohai.prepare(
    step='trainLR-model',
    image='tensorflow/tensorflow:2.6.0',
    default_inputs={
          'dataset': ""
        # 'dataset': 'datum://017f0b6d-387f-5128-b0ac-bac48588135d'
    }
)
with open(valohai.inputs("dataset").path()) as csv_file:
    df = pd.read_csv(csv_file)
print(df.head())

print(df.shape)

print(df.isna().sum())

df1 = df.dropna(subset=["label"])

df1 = df1.dropna(subset=["contrast12"])

print(df1.isna().sum())

print(df1.shape)

print(df1.columns)

# df1.label()

# df1.tail(5)

df1['label'] = (df1.label == "Y").astype(int)

"""## feature engineering

**duplicate removel**
"""

print(df1.columns)

df2 = df1[['contrast11',
           'contrast12', 'contrast13', 'contrast14', 'contrast_avg1', 'contrast21',
           'contrast22', 'contrast23', 'contrast24', 'contrast_avg2',
           'dissimilarity11', 'dissimilarity12', 'dissimilarity13',
           'dissimilarity14', 'dissimilarity_avg1', 'dissimilarity21',
           'dissimilarity22', 'dissimilarity23', 'dissimilarity24',
           'dissimilarity_avg2', 'homogeneity11', 'homogeneity12', 'homogeneity13',
           'homogeneity14', 'homogeneity_avg1', 'homogeneity21', 'homogeneity22',
           'homogeneity23', 'homogeneity24', 'homogeneity_avg_2',
           'energy_feature11', 'energy_feature12', 'energy_feature13',
           'energy_feature14', 'energy_feature_avg1', 'energy_feature21',
           'energy_feature22', 'energy_feature23', 'energy_feature24',
           'energy_feature_avg2', 'correlation_feature11', 'correlation_feature12',
           'correlation_feature13', 'correlation_feature14',
           'correlation_feature_avg1', 'correlation_feature21',
           'correlation_feature22', 'correlation_feature23',
           'correlation_feature24', 'correlation_feature_avg2', 'asm_feature11',
           'asm_feature12', 'asm_feature13', 'asm_feature14', 'asm_feature_avg1',
           'asm_feature21', 'asm_feature22', 'asm_feature23', 'asm_feature24',
           'asm_feature_avg2', 'ALEX_net', 'Cosine similarity', 'td_b7',
           'vents_count_1', 'vents_count_2', 'common_vents_count',
           'similarity_ratio', 'label']]

dp = df2.duplicated()
print(dp.sum())
# df2[dp]

df2.drop_duplicates(inplace=True)
df2.duplicated().sum()

# df1.shape

df2.shape  # after removing duplicate entries, datapoint reduced to half

# figure=df2.boxplot(column='contrast_avg1')

# figure=df2.boxplot(column='contrast_avg2')

# figure=df2.boxplot(column='dissimilarity_avg1')

# figure=df2.boxplot(column='dissimilarity11')

# figure=df2.boxplot(column="asm_feature_avg2")

# figure=df2.boxplot(column='energy_avg2')

"""**Outlier Treatment**"""


# df2.columns

def outliers(ds):
    def outliers_iqr(ys):
        quartile_1, quartile_3 = np.percentile(ys, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((ys > upper_bound) | (ys < lower_bound))

    for i in ds:
        if (ds[i].dtypes == 'float64' or ds[i].dtypes == 'int64'):
            out = list(outliers_iqr(ds[i]))
            if out[0].size:
                print("*****************************************")
                print("Variable \"", i, "\" has following ", len(out[0]), " outliers, which is ",
                      (len(out[0]) / len(ds[i])) * 100, " %.")
                # for j in out[0]:
                #    print("        Outlier value at",j,"th position is", ds[i].loc[j])
                print("Outlier at 5% ", (np.percentile(ds[i], [5]))[0])
                print("Outlier at 95% ", (np.percentile(ds[i], [95]))[0])
    return


outliers(df2)


def outliers_treatment(ds):
    def outliers_iqr(ys):
        quartile_1, quartile_3 = np.percentile(ys, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        return np.where((ys > upper_bound) | (ys < lower_bound))

    for i in ds:
        if (ds[i].dtypes == 'float64' or ds[i].dtypes == 'int64'):
            out = list(outliers_iqr(ds[i]))
            if out[0].size:
                print("*****************************************")
                print("Variable \"", i, "\" has following ", len(out[0]), " outliers, which is ",
                      (len(out[0]) / len(ds[i])) * 100, " %.")
                # for j in out[0]:
                #    print("        Outlier value at",j,"th position is", ds1[i].loc[j])
                print("Outlier at 5% ", (np.percentile(ds[i], [5]))[0])
                print("Outlier at 95% ", (np.percentile(ds[i], [95]))[0])
                if ((len(out[0]) / len(ds[i])) * 100 > 1):
                    lower_bound = (np.percentile(ds[i], [5]))[0]
                    upper_bound = (np.percentile(ds[i], [95]))[0]
                    ds[i][ds[i] <= lower_bound] = lower_bound
                    ds[i][ds[i] >= upper_bound] = upper_bound
                elif ((len(out[0]) / len(ds[i])) * 100 <= 1):
                    median_value = ds[i].median()
                    lower_bound = (np.percentile(ds[i], [5]))[0]
                    upper_bound = (np.percentile(ds[i], [95]))[0]
                    ds[i][ds[i] <= lower_bound] = median_value
                    ds[i][ds[i] >= upper_bound] = median_value

    return


outliers_treatment(df2)

# df2.shape

# df2.columns

# figure=df1['contrast_avg1'].hist(bins=50)
# figure.set_title('contrast_avg1')
# figure.set_xlabel('contrast_avg1')
# figure.set_ylabel('No of image pairs')

# figure=df2['contrast_avg1'].hist(bins=50)
# figure.set_title('contrast_avg1')
# figure.set_xlabel('contrast_avg1')
# figure.set_ylabel('No of image pairs')

# figure=df1['contrast11'].hist(bins=50)
# figure.set_title('contrast11')
# figure.set_xlabel('contrast11')
# figure.set_ylabel('No of image pairs')

# figure=df2['contrast11'].hist(bins=50)
# figure.set_title('contrast11')
# figure.set_xlabel('contrast11')
# figure.set_ylabel('No of image pairs')

# import seaborn as sns
# sns.pairplot(df3,diag_kind='kde')

"""# feature transformation and scaling"""

print(df2.columns)

"""**log transformation**"""

# df3['log_dist'] = np.log(df3['dist']+1)
# df3['log_vents_count_1'] = np.log(df3['vents_count_1']+1)
# df3['log_vents_count_2'] = np.log(df3['vents_count_2']+1)
# df3['log_common_vents_count'] = np.log(df3['common_vents_count']+1)
# df3['log_Vent_Similarity'] = np.log(df3['Vent_Similarity']+1)
# # We created a new column to store the log values

"""**Power Transformer scalar**"""

# figure=df3['log_Vent_Similarity'].hist(bins=50)
# figure.set_title('log_Vent_Similarity')
# figure.set_xlabel('log_Vent_Similarity')
# figure.set_ylabel('No of image pairs')

# figure=df3['log_vents_count_1'].hist(bins=50)
# figure.set_title('log_vents_count_1')
# figure.set_xlabel('log_vents_count_1')
# figure.set_ylabel('No of image pairs')

# figure=df3['log_vents_count_2'].hist(bins=50)
# figure.set_title('log_vents_count_2')
# figure.set_xlabel('log_vents_count_2')
# figure.set_ylabel('No of image pairs')

"""#slicing"""

print(df2.columns)

X = df2[['contrast11', 'contrast12', 'contrast13', 'contrast14', 'contrast_avg1',
         'contrast21', 'contrast22', 'contrast23', 'contrast24', 'contrast_avg2',
         'dissimilarity11', 'dissimilarity12', 'dissimilarity13',
         'dissimilarity14', 'dissimilarity_avg1', 'dissimilarity21',
         'dissimilarity22', 'dissimilarity23', 'dissimilarity24',
         'dissimilarity_avg2', 'homogeneity11', 'homogeneity12', 'homogeneity13',
         'homogeneity14', 'homogeneity_avg1', 'homogeneity21', 'homogeneity22',
         'homogeneity23', 'homogeneity24', 'homogeneity_avg_2',
         'energy_feature11', 'energy_feature12', 'energy_feature13',
         'energy_feature14', 'energy_feature_avg1', 'energy_feature21',
         'energy_feature22', 'energy_feature23', 'energy_feature24',
         'energy_feature_avg2', 'correlation_feature11', 'correlation_feature12',
         'correlation_feature13', 'correlation_feature14',
         'correlation_feature_avg1', 'correlation_feature21',
         'correlation_feature22', 'correlation_feature23',
         'correlation_feature24', 'correlation_feature_avg2', 'asm_feature11',
         'asm_feature12', 'asm_feature13', 'asm_feature14', 'asm_feature_avg1',
         'asm_feature21', 'asm_feature22', 'asm_feature23', 'asm_feature24',
         'asm_feature_avg2', 'ALEX_net', 'Cosine similarity', 'td_b7',
         'vents_count_1', 'vents_count_2', 'common_vents_count',
         'similarity_ratio']]

y = df2['label']
print(y)

##corelation#

import matplotlib.pyplot as plt
import seaborn as sns
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df2.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor["label"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>.3]
relevant_features

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

"""## final spliting and slicing"""

print(df2.label.value_counts())

#4366/720

print(df2.columns)

# X = df2[['vents_count_1', 'vents_count_2', 'common_vents_count',
#        'Vent_Similarity', 'Cosine similarity_b6', 'td_b6',
#        'Cosine similarity_b7', 'td_b7', 'PIQ_Dists']]

# y = df2['label']

X = df2[['contrast11', 'contrast12', 'contrast13', 'contrast14', 'contrast_avg1',
       'contrast21', 'contrast22', 'contrast23', 'contrast24', 'contrast_avg2',
       'dissimilarity11', 'dissimilarity12', 'dissimilarity13',
       'dissimilarity14', 'dissimilarity_avg1', 'dissimilarity21',
       'dissimilarity22', 'dissimilarity23', 'dissimilarity24',
       'dissimilarity_avg2', 'homogeneity11', 'homogeneity12', 'homogeneity13',
       'homogeneity14', 'homogeneity_avg1', 'homogeneity21', 'homogeneity22',
       'homogeneity23', 'homogeneity24', 'homogeneity_avg_2',
       'energy_feature11', 'energy_feature12', 'energy_feature13',
       'energy_feature14', 'energy_feature_avg1', 'energy_feature21',
       'energy_feature22', 'energy_feature23', 'energy_feature24',
       'energy_feature_avg2', 'correlation_feature11', 'correlation_feature12',
       'correlation_feature13', 'correlation_feature14',
       'correlation_feature_avg1', 'correlation_feature21',
       'correlation_feature22', 'correlation_feature23',
       'correlation_feature24', 'correlation_feature_avg2', 'asm_feature11',
       'asm_feature12', 'asm_feature13', 'asm_feature14', 'asm_feature_avg1',
       'asm_feature21', 'asm_feature22', 'asm_feature23', 'asm_feature24',
       'asm_feature_avg2', 'ALEX_net', 'Cosine similarity', 'td_b7',
       'vents_count_1', 'vents_count_2', 'common_vents_count',
       'similarity_ratio']]

y = df2['label']

from sklearn.model_selection import train_test_split
from collections import Counter

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)
print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")

# from imblearn.under_sampling import RandomUnderSampler
# under_sampler = RandomUnderSampler(random_state=42)
# X_sample, y_sample = under_sampler.fit_resample(X_train, y_train)
# print(f"Training target statistics: {Counter(y_sample)}")
# print(f"Testing target statistics: {Counter(y_test)}")

"""# Logistic regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

#base model

logit_model=LogisticRegression()
logit_model.fit(X_train,y_train)
y_pred=logit_model.predict(X_test)

print('Precision score %s' % precision_score(y_test, y_pred))
print('Recall score %s' % recall_score(y_test, y_pred))
print('F1-score score %s' % f1_score(y_test, y_pred))
print('Accuracy score %s' % accuracy_score(y_test, y_pred))
print('confusion matrix %s' % confusion_matrix(y_test, y_pred))

#model1 with class weight

logit_model=LogisticRegression(class_weight={0:1,1:6})
logit_model.fit(X_train,y_train)
y_pred_weight=logit_model.predict(X_test)

print('Precision score %s' % precision_score(y_test, y_pred_weight))
print('Recall score %s' % recall_score(y_test, y_pred_weight))
print('F1-score score %s' % f1_score(y_test, y_pred_weight))
print('Accuracy score %s' % accuracy_score(y_test, y_pred_weight))
print('confusion matrix %s' % confusion_matrix(y_test, y_pred_weight))

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# define hyperparameters
w = [{0:1,1:5}]
crange = np.arange(0.1, 5, 0.1)
crange = [100, 10, 1.0, 0.1,0.5, 0.01]
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
hyperparam_grid = {"class_weight":w,
                   "penalty": ["l1", "l2"],
                   "C": crange,
                   "solver":solver,
                   "fit_intercept": [True, False] }

# logistic model classifier

cv=KFold(n_splits=5,random_state=2018,shuffle=True)
lg = LogisticRegression(random_state=2018)
# define evaluation procedure
scoring = ['accuracy','recall']
grid = GridSearchCV(lg,hyperparam_grid,scoring=scoring, cv=cv, n_jobs=-1, refit='recall')
grid.fit(X_train,y_train)
print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

# final model with selected parameters

logit_model=LogisticRegression(C =0.5,penalty = "l1",fit_intercept= True,class_weight={0:1, 1:6},solver = 'liblinear',random_state=2018)
logit_model.fit(X_train,y_train)
y_pred=logit_model.predict(X_test)

#################################################################
print('Precision score %s' % precision_score(y_test, y_pred))
print('Recall score %s' % recall_score(y_test, y_pred))
print('F1-score score %s' % f1_score(y_test, y_pred))
print('Accuracy score %s' % accuracy_score(y_test, y_pred))
print('confusion matrix %s' % confusion_matrix(y_test, y_pred))
#################################################################

import joblib


# Save the model as a pickle in a file
output_path = valohai.outputs().path(f'logistic_regression.pkl')
joblib.dump(logit_model, output_path)
#joblib.dump(logit_model, pkl_data+'/logistic_regression.pkl')

"""**logistic regression recall - 81 and acc - 71**

***Optimal threshold setting***
"""

# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score
#
# y_pred_prob=logit_model.predict_proba(X_test)[:,1]
#
# #### Calculate the ROc Curve
#
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# thresholds
#
# from sklearn.metrics import accuracy_score,recall_score
#
# accuracy_ls = []
# recall_ls = []
# for thres in thresholds:
#     y_pred = np.where(y_pred_prob>thres,1,0)
#     accuracy_ls.append(accuracy_score(y_test, y_pred, normalize=True))
#     recall_ls.append(recall_score(y_test, y_pred))
# accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],
#                         axis=1)
# accuracy_ls.columns = ['thresholds','accuracy']
# accuracy_ls['recall'] = recall_ls
#
# # # accuracy_ls.sort_values(by='recall', ascending=False, inplace=True)
# output_path = valohai.outputs().path(f't.csv')
# joblib.dump(logit_model, output_path)
# #accuracy_ls.to_csv(pkl_data+'t.csv')
#
# """**logisticregression with undesampler**"""
#
# logit_model=LogisticRegression(C = 1,penalty = "l2",fit_intercept= True,solver = 'lbfgs',random_state=2018)
# logit_model.fit(X_sample, y_sample)
# y_pred=logit_model.predict(X_test)
#
# print('Precision score %s' % precision_score(y_test, y_pred))
# print('Recall score %s' % recall_score(y_test, y_pred))
# print('F1-score score %s' % f1_score(y_test, y_pred))
# print('Accuracy score %s' % accuracy_score(y_test, y_pred))
# print('confusion matrix %s' % confusion_matrix(y_test, y_pred))
