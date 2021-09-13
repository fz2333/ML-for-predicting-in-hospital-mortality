import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import shap
import lime
import lime.lime_tabular

# data = pd.read_csv('./Data2.23.csv')
data = pd.read_csv(r'C:\Users\Administrator\Desktop\gt\7.28\Data2.23.csv')
np.set_printoptions(suppress=True)
y = data["Mortality"].values
X = data.iloc[:, 0:41].values

skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
lr_model = XGBClassifier()
# lr_model = LogisticRegression()  # 调用模型，但是并未经过任何调参操作，使用默认值
# lr_model = tree.DecisionTreeClassifier()
# lr_model = KNeighborsClassifier()
# lr_model = GaussianNB()
# lr_model = MultinomialNB()

col = ['Gender', 'Age', 'Height', 'Weight', 'Body mass index', 'Systolic blood pressure',
       'Diastole blood pressure', 'Hypertension', 'Diabetes', 'Stroke', 'Atherosclerosis',
       'Marfan Syndrome', 'Smoking', 'Drinking', 'Symptom', 'Type of AAD(Stanford)',
       'Treatment',
       'White blood cells count', 'Lymphocyte Ratio', 'Neutrophil Ratio', 'Platelet count',
       'Hemoglobin', 'Alanine transaminase', 'Aspartate aminotransferase', 'Albumin',
       'Total bilirubin',
       'Direct bilirubin', 'Creatinine', 'Blood urea nitrogen', 'Uric acid', 'Myoglobin',
       'Creatine kinase', 'Creatine kinase-MB', 'Troponin T', 'B-type natriuretic peptide',
       'D-Dimer', 'Ischemia-modified albumin', 'C-reactive protein',
       'Erythrocyte Sedimentation Rate', 'Procalcitonin',
       'Lactate Dehydrogenase']

importance = np.zeros(shape=(1344, 41))
for train, test in skf.split(X, y):
    lr_model.fit(X[train], y[train])

    explainer = shap.TreeExplainer(lr_model)
    # explainer = shap.KernelExplainer(lr_model.predict_proba, X[train])
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, df, plot_type="bar")

    # shapV = shap_values[0]
    # shapV = shap_values[1]
    for i in range(0, 1344):
        for j in range(0, 41):
            importance[i][j] += shap_values[i][j]
            # importance[i][j] += shapV[i][j]

    # LIME ed by Yuankai 08/04/21
    visual_ind = 5  # TP fold1 5 21 22 26 29 32 34 35 39 50 51 64 70 71 72 93 103
    # visual_ind = 0  # TN 我们看第一个病人，可以看任意病人
    #FN fold1 13 20 40 63 112 115 118
    # visual_ind = 40  # FN fold2 35 38 39 69 82 92 95 110 117
    # visual_ind = 2  # FP

    test_pred = lr_model.predict(X[test])
    print(test_pred)
    print(y[test])
    print(test)

    a = y[test]
    for i in range(0, len(test_pred)):
        if(test_pred[i]==1 and a[i]==0):
            print(i)

    errors = test_pred - y[test]
    sorted_errors = np.argsort(abs(errors))

    train_X = X[train]
    test_X = X[test]
    explainer2 = lime.lime_tabular.LimeTabularExplainer(train_X, feature_names=col, class_names=['Mortality'],
                                                        verbose=True, mode='regression')
    print('Error =', errors[visual_ind])
    exp = explainer2.explain_instance(test_X[0], lr_model.predict, num_features=41)
    # exp.show_in_notebook(show_table=True)
    exp.as_pyplot_figure()
    # a = 1

for i in range(0, 1344):
    for j in range(0, 41):
        importance[i][j] = importance[i][j] / 10

df = pd.DataFrame(X, columns=col)
# shap.summary_plot(importance, df)
shap.summary_plot(importance, df, plot_type="bar")

# shap.dependence_plot("Management", shap_values, df, interaction_index=None)
# shap.dependence_plot("Type of AAD(Stanford)", shap_values, df, interaction_index=None)
# shap.dependence_plot("Ischemia-modified albumin", shap_values, df, interaction_index=None)
# shap.dependence_plot("C-reactive protein", shap_values, df, interaction_index=None)
# shap.force_plot(explainer.expected_value, shap_values, df)

#
# data1 = pd.DataFrame(importance)
# data1.to_csv(r"C:\Users\Administrator\Desktop\gt\7.28\xgbShapValue.csv", header=False, index=False)

