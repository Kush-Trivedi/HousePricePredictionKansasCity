import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import ensemble

df = pd.read_csv("kc_house.csv")

print(df.head(5))
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())

print(df.bedrooms.value_counts())
sns.countplot(x=df.bedrooms)
plt.show()

print(df.floors.value_counts())
sns.countplot(x=df.floors)
plt.show()

print(df.grade.value_counts())
sns.countplot(x=df.grade)
plt.show()

sns.boxplot(x=df.grade, y=df.price)
plt.show()

sns.regplot(x=df.grade, y=df.price)
plt.show()

sns.displot(x=df.price)
plt.show()

df['price'] = df['price'].apply(lambda x: np.log(x))
sns.displot(x=df.price)
plt.show()

continuous_columns = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront',
                      'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
                      'sqft_living15', 'sqft_lot15']
for i in continuous_columns:
    print("Column Name:", i)
    sns.displot(x=df[i])
    plt.show()

for i in continuous_columns:
    print("Column Name:", i)
    sns.boxplot(x=df[i])
    plt.show()

mod = []


def find_outlier(df_in, column_name):
    q1 = df_in[column_name].quantile(0.25)
    q3 = df_in[column_name].quantile(0.75)
    iqr = q3 - q1
    fence_high = q3 + 1.5 * iqr
    df_high = df_in.loc[(df_in[column_name] > fence_high)]
    outlier_percentage = (df_high.shape[0] / len(df)) * 100
    outlier_percentage = round(outlier_percentage, 2)
    print(outlier_percentage)
    mod.append((column_name, outlier_percentage))


print(mod)

for i in continuous_columns:
    find_outlier(df, i)

length = len(continuous_columns)
result = []
names = []
for i in range(0, length - 1):
    result.append(mod[i][1])
    names.append(mod[i][0])

print(names)

pData = pd.DataFrame(result)
pData.rename(columns={0: "outlier_percentage"}, inplace=True)

pData.plot.bar(figsize=(20, 8))
plt.xticks(np.arange(length - 1), names)
plt.title("Percentage Outliers for each Column")
plt.axhline(5)
plt.show()

df.drop(['date'], axis=1, inplace=True)
print(df.corr())
print(df.cov())

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.heatmap(df.corr(), annot=True, linewidths=0.2, vmax=0.9)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

y = df['price']
X = df.drop('price', axis=1)
xc = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(np.abs(round(stats.norm.isf(q=0.025), 2)))

lin_reg = LinearRegression()
model = lin_reg.fit(X_train, y_train)
print(f'R^2 score for train: {lin_reg.score(X_train, y_train)}')
print(f'\nR^2 score for test: {lin_reg.score(X_test, y_test)}')

print(df.mean(), np.std(df, ddof=1))

Xc = sm.add_constant(X)
lin_reg = sm.OLS(y, Xc).fit()
print(lin_reg.summary())

lin_reg = LinearRegression()
lin_reg.fit(X, y)

print(f'Coefficients: {lin_reg.coef_}')
print(f'\nIntercept: {lin_reg.intercept_}')
print(f'\nR^2 score: {lin_reg.score(X, y)}')

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)
model.fit(X_train, y_train)
model_score = model.score(X_train, y_train)
print('R2 sq: ', model_score)
y_predicted = model.predict(X_test)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_predicted))
print('Test Variance score: %.2f' % r2_score(y_test, y_predicted))
print("RMSE:%.2f" % np.sqrt(mean_squared_error(y_test, y_predicted)))
print(f'R^2 score for train: {model.score(X_train, y_train)}')
print(f'\nR^2 score for test: {model.score(X_test, y_test)}')

fig, ax = plt.subplots()
ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Actual vs Predicted")
plt.show()
