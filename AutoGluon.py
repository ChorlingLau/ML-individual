from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

output_path = "submission.txt"

data_train = TabularDataset("data/train.csv")
data_test = TabularDataset("data/test.csv")
all_features = pd.concat((data_train.iloc[:, 1:-1], data_test.iloc[:, 1:]))
all_features = pd.get_dummies(all_features)
drop_list = ['sex_F', 'address_R', 'famsize_LE3', 'Pstatus_A']
for feature in all_features:
    if "no" in feature:
        drop_list.append(feature)
all_features = pd.get_dummies(all_features).drop(columns=drop_list)
n_train = data_train.shape[0]
G3 = data_train.G3
data_train = all_features[:n_train]
data_test = all_features[n_train:]
data_train = pd.concat((data_train, G3.T), axis=1)

predictor = TabularPredictor(label='G3', problem_type='regression').fit(train_data=data_train)

print(predictor.feature_importance(data_train))

pred = predictor.predict(data_test)
fout = open(output_path, "w", encoding='utf-8')
for i in pred:
    fout.write(f"{round(i)}\n")
