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

fig_na, ax = plt.subplots(figsize=(10, 6))
msno.matrix(train_df, figsize=(12,5), ax=ax)
print("=====Missing Rate=====")
for col in train_df.columns:
    missing_ratio = train_df[col].isnull().sum() / len(train_df[col])
    print(f"{col}: {round(missing_ratio, 2)}%")

fig_dist, ax = plt.subplots(figsize=(10,6))
sns.histplot(train_df["count"], kde=True, stat="density", ax=ax)

print(f'Skewness: {train_df["count"].skew()}')
print(f'Kurtosis: {train_df["count"].kurt()}')


#%% eda
train_df["year"] = train_df["datetime"].dt.year
train_df["month"] = train_df["datetime"].dt.month
train_df["day"] = train_df["datetime"].dt.day
train_df["hour"] = train_df["datetime"].dt.hour
train_df["dayofweek"] = train_df["datetime"].dt.dayofweek
train_df["year_month"] = train_df["datetime"].map(lambda x: f"{x.year}-{x.month}")


fig_bar, axes = plt.subplots(nrows = 2, ncols = 2)
fig_bar.set_size_inches(18,10)

sns.barplot(data=train_df, x = "year", y = "count", ax = axes[0][0])
sns.barplot(data=train_df, x = "month", y = "count", ax = axes[0][1])
sns.barplot(data=train_df, x = "day", y = "count", ax = axes[1][0])
sns.barplot(data=train_df, x = "hour", y = "count", ax = axes[1][1])

axes[0][0].set(ylabel = "count", title = "Rental amount by year")
axes[0][1].set(ylabel = "count", title = "Rental amount by month")
axes[1][0].set(ylabel = "count", title = "Rental amount by day")
axes[1][1].set(ylabel = "count", title = "Rental amount by hour")


fig_point, axes = plt.subplots(nrows=6)
fig_point.set_size_inches(18, 25)

sns.pointplot(train_df, x="hour", y="count", ax=axes[0])
sns.pointplot(train_df, x="hour", y="count", hue="workingday", ax=axes[1])
sns.pointplot(train_df, x="hour", y="count", hue="dayofweek", ax=axes[2])
sns.pointplot(train_df, x="hour", y="count", hue="weather", ax=axes[3])
sns.pointplot(train_df, x="hour", y="count", hue="season", ax=axes[4])
plt.sca(axes[5])
plt.xticks(rotation=30, ha="right")
sns.pointplot(train_df, x="year_month", y="count", ax=axes[5])


corr_data = train_df[["temp", "atemp", "humidity", "windspeed", "count"]]
fig_corr, ax = plt.subplots(figsize=(12,10))
plt.title("Correlation of Numeric Feautures with Rental Count")
sns.heatmap(corr_data.corr(), annot=True, cmap=plt.cm.PuBu)


fig_scatter, axes = plt.subplots(ncols=3, figsize=(12,5))
sns.scatterplot(x='temp', y='count', data=train_df, ax=axes[0])
sns.scatterplot(x='windspeed', y='count', data=train_df, ax=axes[1])
sns.scatterplot(x='humidity', y='count', data=train_df, ax=axes[2])


for col in train_df[["temp", "humidity", "windspeed"]]:
    print(f'{col} - Skewness: {train_df["count"].skew()} / Kurtosis: {train_df["count"].kurt()}')

fig_box, axes = plt.subplots(nrows=4, ncols=2, figsize=(17,18))
sns.boxplot(data=train_df, y="count", x="season", ax=axes[0][0])
sns.boxplot(data=train_df, y="count", x="weather", ax=axes[0][1])
sns.boxplot(data=train_df, y="count", x="holiday", ax=axes[1][0])
sns.boxplot(data=train_df, y="count", x="workingday", ax=axes[1][1])
sns.boxplot(data=train_df, y="count", x="dayofweek", ax=axes[2][0])
sns.boxplot(data=train_df, y="count", x="year", ax=axes[2][1])
sns.boxplot(data=train_df, y="count", x="month", ax=axes[3][0])
sns.boxplot(data=train_df, y="count", x="hour", ax=axes[3][1])

#%% feature engineering
fig_dist
train_df["count_log"] = train_df["count"].map(lambda x: np.log(x) if x>0 else 0)

fig_dist_log, ax = plt.subplots(figsize=(10,6))
sns.histplot(train_df["count_log"], kde=True, stat="density", ax=ax)

print(f'Skewness: {train_df["count_log"].skew()}')
print(f'Kurtosis: {train_df["count_log"].kurt()}')

fig_corr_log, ax = plt.subplots(figsize=(10,6))
sns.heatmap(data=train_df[[
    "season", "holiday", "workingday", "weather", "temp", "humidity", "windspeed", "count_log"
    ]].corr(), annot=True, square=True, ax=ax)

train_df = pd.get_dummies(train_df, columns=["weather"], prefix="weather")
train_df = pd.get_dummies(train_df, columns=["season"], prefix="season")

train_df.columns
X = train_df[[
    "holiday", "workingday", "temp", "humidity", "windspeed",
    "weather_1", "weather_2", "weather_3", "weather_4", 
    "season_1", "season_2", "season_3", "season_4", 
    ]]
Y = train_df["count_log"]


test_df = pd.get_dummies(test_df, columns=["weather"], prefix="weather")
test_df = pd.get_dummies(test_df, columns=["season"], prefix="season")

test_X = test_df[[
    "holiday", "workingday", "temp", "humidity", "windspeed",
    "weather_1", "weather_2", "weather_3", "weather_4", 
    "season_1", "season_2", "season_3", "season_4", 
    ]]

X.describe()
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
for max_depth in tqdm(range(2, 10)):
    for n_estimators in [50, 100, 150]:
        metrics = []
        for seed in range(10):
            train_x, valid_x, train_y, valid_y = train_test_split(X, Y, random_state=seed)

            gboost = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators)
            gboost.fit(train_x, train_y)
            pred = gboost.predict(valid_x)
            metric = calculate_metric(np.exp(pred), np.exp(valid_y))
            metrics.append(metric)
        
        mean_metrics = np.mean(metrics)
        res[mean_metrics] = [max_depth, n_estimators]
        
metrics = list(res.keys())
metric = min(metrics)
optimal_max_depth, optimal_n_estimators = res.get(metric)

gboost = GradientBoostingRegressor(max_depth=optimal_max_depth, n_estimators=optimal_n_estimators)
gboost.fit(X, Y)
pred_Y = gboost.predict(test_X)
submit_df["count"] = np.exp(pred_Y)
submit_df.to_csv("/Users/wonhyung64/Downloads/submission.csv", index=False)
