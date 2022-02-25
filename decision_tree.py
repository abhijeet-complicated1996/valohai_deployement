import numpy as np
import valohai
import matplotlib.pyplot as plt
import pandas as pd


valohai.prepare(
    step='trainDT-model',
    image='tensorflow/tensorflow:2.6.0',
    default_inputs={
        'dataset': 'datum://017f1b1c-d53c-2a3a-6208-11d2cd096c17'
    }
)
with open(valohai.inputs("dataset").path()) as csv_file:
    df2 = pd.read_csv(csv_file)

df2.label.value_counts()
df2.columns
print(df2)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
dt_y_pred = dt_model.predict(X_test)
print(dt_y_pred)

#print('Precision score %s' % precision_score(y_test, dt_y_pred) average='micro')
#print('precision_score %s' % precision_score(y_train, dt_y_pred))
# print('Recall score %s' % recall_score(y_test, dt_y_pred))
# print('F1-score score %s' % f1_score(y_test, dt_y_pred))
# print('Accuracy score %s' % accuracy_score(y_test, dt_y_pred))
print("Recall Score : ",recall_score(y_test, dt_y_pred,
                                           pos_label='positive',
                                           average='micro'))
print("Precision Score : ",precision_score(y_test, dt_y_pred,
                                           pos_label='positive',
                                           average='weighted'))
# print("accuracy score : ",accuracy score(y_test, dt_y_pred,
#                                            pos_label='positive',
#                                            average='weighted'))
# print("F1-score : ",F1-score(y_test, dt_y_pred,
#                                            pos_label='positive',
#                                            average='weighted'))

import joblib

# Save the model as a pickle in a file
output_path = valohai.outputs().path(f'decision_tree_97.pkl')
joblib.dump(dt_model, output_path)
#joblib.dump(dt_model, '/content/decision_tree_97.pkl')

