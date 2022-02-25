# vh yaml step test.py
import numpy as np
import pandas as pd
import valohai


default_inputs = {
    'myinput': 'datum://017f0b6d-387f-5128-b0ac-bac48588135d'
}

# Create a step 'train' in valohai.yaml with a set of inputs
valohai.prepare(step="preprocess-data", image="tensorflow/tensorflow:2.6.1-gpu", default_inputs=default_inputs)

# Open the CSV file from sValohai inputs
with open(valohai.inputs("myinput").path()) as csv_file:
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

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from collections import Counter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)
print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")

from imblearn.under_sampling import RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)
X_sample, y_sample = under_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_sample)}")
print(f"Testing target statistics: {Counter(y_test)}")


out_path = valohai.outputs().path('preprocessed_data.csv')
df2.to_csv(out_path)



