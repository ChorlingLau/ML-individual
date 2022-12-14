import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

df = pd.read_csv('data/train.csv')
print(df.head())
print(df.shape)

# sns.displot(df.G3)
# plt.show()

# continuous_features = df.select_dtypes("int")
# f, ax = plt.subplots(5, 3, figsize=(25, 40))
# for i, feature in enumerate(continuous_features):
#     sns.histplot(continuous_features[feature], ax=ax[i // 3, i % 3], element="step")
# plt.show()
# for i, feature in enumerate(continuous_features):
#     sns.histplot(np.log1p(continuous_features[feature]), ax=ax[i // 3, i % 3], element="step")
# plt.show()
# for i, feature in enumerate(continuous_features):
#     if feature == 'school' or feature == 'age' or feature == 'absences':
#         continue
#     print(feature)
#     sns.histplot(np.expm1(continuous_features[feature]), ax=ax[i // 3, i % 3], element="step")
# plt.show()

# corr = df.corr()
# plt.subplots(figsize=(10, 8))
# sns.heatmap(corr, square=True)
# plt.show()

# k = 15
# cols = corr.nlargest(k, "G3").G3.index
# cm = np.corrcoef(df[cols].values.T)
# plt.subplots(figsize=(15, 12))
# sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f", yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

y = df['G3']
df = df.drop(columns=['school', 'G3'])
df = pd.get_dummies(df)
drop_list = ['sex_F', 'address_R', 'famsize_LE3', 'Pstatus_A']
for feature in df:
    if "no" in feature:
        drop_list.append(feature)
x = df.drop(columns=drop_list)
print(x.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# model = LGBMRegressor()
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
# _y_pred = np.array([y if y > 0 else 0 for y in y_pred])
#
# a = pd.DataFrame()
# a['预测值'] = list(_y_pred)
# a['实际值'] = list(y_test)
# print(a.head())

# rmse = np.sqrt(((_y_pred - y_test) ** 2).mean())
# print("RMSE:", rmse)

# parameters = {
#     'num_leaves': [10, 20, 30, 40, 50],
#     'n_estimators': [20, 40, 60, 80],
#     'learning_rate': [1e-5, 1e-3, 1e-1],
# }

# parameters = {
#     'num_leaves': [i for i in range(3, 8)],
#     'n_estimators': [2*i for i in range(3, 8)],
#     'learning_rate': [0.1*i for i in range(1, 11)],
# }

parameters = {
    'num_leaves': [4],
    'n_estimators': [i for i in range(12, 16)],
    'learning_rate': [0.01*i+0.3 for i in range(-9, 10)],
}

model = LGBMRegressor()
estimator = GridSearchCV(model, parameters, scoring='neg_root_mean_squared_error', cv=5)
estimator.fit(x, y)
print(estimator.best_params_)
print(estimator.best_score_)
