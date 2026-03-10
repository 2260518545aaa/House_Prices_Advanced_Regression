import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import xgboost as xgb 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge , ElasticNet
from sklearn.ensemble import RandomForestRegressor

# 1. Data Preprocessing
# import data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# separate x and y variables
train_y = train_data['SalePrice']
train_x = train_data.drop(["Id", "SalePrice"], axis = 1)
test_x = test_data.drop(["Id"],axis = 1)

# fill in missing values and standardize numerical variables
df_x = pd.concat([train_x, test_x], axis = 0)
num_cols_x = df_x.select_dtypes(include = np.number).columns
df_x[num_cols_x] = df_x[num_cols_x].fillna(df_x[num_cols_x].median())
scaler = StandardScaler()  # prepare standardization tool
df_x[num_cols_x] = scaler.fit_transform(df_x[num_cols_x]) 

# fill in missing values for categorical variables 
# change to numeric variables( feature engineering: one-hot encoding)
cat_cols_x = df_x.select_dtypes(include = 'object').columns
df_x[cat_cols_x] = df_x[cat_cols_x].fillna('Unknown')
df_x = pd.get_dummies(df_x, columns = cat_cols_x)

# split the data back into train and test sets
train_x = df_x[:len(train_x)].reset_index(drop = True)
test_x = df_x[len(train_x):].reset_index(drop = True)

# 2. EDA
# check data info and description
print("Data Info:")
print(train_x.info(), train_y.info())
print("\nData Description:")
print(train_x.describe(), train_y.describe(), "\n")

# SalePrice distribution and outliers
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)  # 2 row and 1 columns, 1st position
sns.histplot(train_y)
plt.title("Distribution of SalePrice")
plt.subplot(2, 1, 2)  # 2nd position
sns.boxplot(y=train_y)
plt.title("Boxplot of SalePrice")
plt.tight_layout()  # auto adjust subplot intervals
plt.show()

# missing value heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(train_data.drop(["Id", "SalePrice"], axis=1).isnull(), 
            cbar=True, yticklabels=False, cmap='Reds')
plt.title("NaN value in train_df disstribution heatmap(1=has NaN)",
           fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Entries", fontsize=12)
plt.tight_layout()
plt.show()

# 3. Model Training and predict
# model 1: Linear Regression
model1 = LinearRegression().fit(train_x, train_y) 
pre1_y = model1.predict(test_x)

submit1 = pd.DataFrame({"Id": test_data["Id"], "SalePrice": pre1_y})
submit1.to_csv("data/my_data_LinearRegression.csv", index = False)

# model 2: Random Forest Regression
model2 = RandomForestRegressor().fit(train_x, train_y)
pre2_y = model2.predict(test_x)

submit2 = pd.DataFrame({"Id": test_data["Id"], "SalePrice": pre2_y})
submit2.to_csv("data/my_data_RandomForestRegression.csv", index = False)

# Regularize Linear Regression(prevent overfitting)
# model 3: Lasso regression
# some features may not important, let its coefficients become 0
# can auto select important features
model3 = Lasso(alpha=0.001, max_iter=10000).fit(train_x, train_y)
pre3_y = model3.predict(test_x)

submit3 = pd.DataFrame({"Id": test_data["Id"], "SalePrice": pre3_y})
submit3.to_csv("data/my_data_LassoRegression.csv", index = False)

# model 4: Ridge regression
#  keep all features, small all coefficients, but not 0
model4 = Ridge(alpha=1.0).fit(train_x, train_y)
pre4_y = model4.predict(test_x)

submit4 = pd.DataFrame({"Id": test_data["Id"], "SalePrice": pre4_y})
submit4.to_csv("data/my_data_RidgeRegression.csv", index = False)

# model 5: ElasticNet regression
# Lasso + Ridge, can both select features and keep all features, but not 0
model5 = ElasticNet(alpha=0.001, l1_ratio=0.5).fit(train_x, train_y) 
pre5_y = model5.predict(test_x)

submit5 = pd.DataFrame({"Id": test_data["Id"], "SalePrice": pre5_y})
submit5.to_csv("data/my_data_ElasticNetRegression.csv", index = False) 

# model 6: XGBoost regression
# extreme gradient boosting
# handle NaN/outlier/important features/non-linear relationships
model6 = xgb.XGBRegressor().fit(train_x, train_y)
pre6_y = model6.predict(test_x) 

submit6 = pd.DataFrame({"Id": test_data["Id"], "SalePrice": pre6_y})
submit6.to_csv("data/my_data_XGBoostRegression.csv", index = False) 