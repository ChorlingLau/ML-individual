import pandas as pd
from lightgbm import LGBMRegressor

output_path = "submission.txt"

df = pd.read_csv('data/train.csv')

y = df['G3']
df = df.drop(columns=['school', 'G3'])
df = pd.get_dummies(df)
drop_list = ['sex_F', 'address_R', 'famsize_LE3', 'Pstatus_A']
for feature in df:
    if "no" in feature:
        drop_list.append(feature)
x = df.drop(columns=drop_list)

model = LGBMRegressor(
    num_leaves=4,
    n_estimators=12,
    learning_rate=0.39,
)
# model = LGBMRegressor(
#     num_leaves=4,
#     n_estimators=14,
#     learning_rate=0.27,
# )
# model = LGBMRegressor(
#     num_leaves=4,
#     n_estimators=8,
#     learning_rate=0.3,
# )
model.fit(x, y)

df = pd.read_csv('data/test.csv')
df = df.drop(columns=['school'])
df = pd.get_dummies(df)
drop_list = ['sex_F', 'address_R', 'famsize_LE3', 'Pstatus_A']
for feature in df:
    if "no" in feature:
        drop_list.append(feature)
x = df.drop(columns=drop_list)
y_pred = model.predict(x)

fout = open(output_path, "w", encoding='utf-8')
for i in y_pred:
    fout.write(f"{round(i)}\n")
