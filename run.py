#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import missingno as msno
from tqdm import tqdm

# %% data import, check
data_dir = "/Users/wonhyung64/data/bike_demand"
os.listdir(data_dir)
train_df = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["datetime"])
test_df = pd.read_csv(f"{data_dir}/test.csv", parse_dates=["datetime"])
submit_df = pd.read_csv(f"{data_dir}/sampleSubmission.csv")

train_df.head()
test_df.head()

train_df.info()
test_df.info()

msno.matrix(train_df, figsize=(12,5))
print("=====Missing Rate=====")
for col in train_df.columns:
    missing_ratio = train_df[col].isnull().sum() / len(train_df[col])
    print(f"{col}: {round(missing_ratio, 2)}%")

fig, ax = plt.subplots(1, 1, figsize=(10,6))
sns.histplot(train_df["count"], kde=True, stat="density", ax=ax)

print(f'Skewness: {train_df["count"].skew()}')
print(f'Kurtosis: {train_df["count"].kurt()}')

train_df.describe()


#%% eda
train_df["year"] = train_df["datetime"].dt.year
train_df["month"] = train_df["datetime"].dt.month
train_df["day"] = train_df["datetime"].dt.day
train_df["hour"] = train_df["datetime"].dt.hour
train_df["dayofweek"] = train_df["datetime"].dt.dayofweek

fig1, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows = 2, ncols = 2)
fig1.set_size_inches(18,10)

sns.barplot(data=train_df, x = "year", y = "count", ax = ax1)
sns.barplot(data=train_df, x = "month", y = "count", ax = ax2)
sns.barplot(data=train_df, x = "day", y = "count", ax = ax3)
sns.barplot(data=train_df, x = "hour", y = "count", ax = ax4)

ax1.set(ylabel = "count", title = "Rental amount by year")
ax2.set(ylabel = "count", title = "Rental amount by month")
ax3.set(ylabel = "count", title = "Rental amount by day")
ax4.set(ylabel = "count", title = "Rental amount by hour")

fig2, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5)
fig2.set_size_inches(18, 25)

sns.pointplot(train_df, x="hour", y="count", ax=ax1)
sns.pointplot(train_df, x="hour", y="count", hue="workingday", ax=ax2)
sns.pointplot(train_df, x="hour", y="count", hue="dayofweek", ax=ax3)
sns.pointplot(train_df, x="hour", y="count", hue="weather", ax=ax4)
sns.pointplot(train_df, x="hour", y="count", hue="season", ax=ax5)

corr_data = train_df[["temp", "atemp", "humidity", "windspeed", "count"]]


fig3, ax = plt.subplots(figsize=(12,10))
plt.title("Correlation of Numeric Feautures with Rental Count")
sns.heatmap(corr_data.corr(), annot=True, cmap=plt.cm.PuBu)

#%% preprocessing
X = train_df.iloc[:, 1:-3]
Y = train_df.loc[:,"count"]
test_X = test_df.iloc[:, 1:]

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
test_X = scaler.transform(test_X)

#%% modeling
def calculate_metric(pred, true):
    n = np.shape(pred)[0]
    metric = np.sqrt(
        np.sum(
            np.square(np.log(np.clip(pred + 1, 1e-9, None)) - np.log(np.clip(true + 1, 1e-9, None)))
            ) / n
        )

    return metric


res = {}
for max_depth in tqdm(range(1, 30)):
    metrics = []
    for seed in range(10):
        train_x, valid_x, train_y, valid_y = train_test_split(X, Y, random_state=seed)

        gboost = GradientBoostingRegressor(max_depth=max_depth)
        gboost.fit(train_x, train_y)
        pred = gboost.predict(valid_x)
        metric = calculate_metric(pred, valid_y)
        metrics.append(metric)
    
    mean_metrics = np.mean(metrics)
    res[mean_metrics] = max_depth
        
metrics = list(res.keys())
metric = min(metrics)
optimal_max_depth = res.get(metric)

gboost = GradientBoostingRegressor(max_depth=optimal_max_depth)
gboost.fit(X, Y)
pred_Y = gboost.predict(test_X)
submit_df["count"] = pred_Y
submit_df.to_csv("/Users/wonhyung64/Downloads/submission.csv", index=False)

os.getcwd()




# %%
