# 0 Import modulus
import pandas as pd
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# 1 Import database
data = pd.read_excel() # Path of the database (Database.xlsx)
X = data.iloc[1:366, 0:8].values # Independent variable
y = data.iloc[1:366, 9:10].values # Dependent variable

# 2 Set training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1009) # 80% for training set and 20% for testing set

# 3 XGBoost prediction
n_e,m_d,l_r =80, 15, 0.075 # Hyperparameters
model_r = XGBR(n_estimators=n_e,max_depth=m_d,learning_rate=l_r,reg_lambda=1.5,reg_alpha=1.3).fit(X_train,y_train) # Import XGBoost
y_pre = model_r.predict(X_test) # Predictive value of testing set
y_pre_t=model_r.predict(X_train) # Predictive value of training set

# 4 Output
print(f"R-square of testing set: {r2_score(y_test, y_pre)}")
print(f"R-square of training set: {r2_score(y_train, y_pre_t)}")
print(f"RMSE of testing set: {mean_squared_error(y_test, y_pre)**0.5}")
print(f"RMSE of training set: {mean_squared_error(y_train, y_pre_t)**0.5}")
print(f"MAPE of testing set: {mean_absolute_percentage_error(y_test, y_pre)}")
print(f"MAPE of training set: {mean_absolute_percentage_error(y_train, y_pre_t)}")
# Generate data file for the testing set
y_pre = np.array(y_pre).T
y_test = np.squeeze(y_test)
y_test = np.array(y_test).T
scatter = pd.DataFrame()
scatter["y_pre"] = pd.Series(y_pre)
scatter["y_test"] = pd.Series(y_test)
scatter.to_excel("./xgb_test.xlsx", index=False)
# Generate data file for the training set
y_pre_t = np.array(y_pre_t).T
y_train = np.squeeze(y_train)
y_train = np.array(y_train).T
scatter = pd.DataFrame()
scatter["y_pre_t"] = pd.Series(y_pre_t)
scatter["y_train"] = pd.Series(y_train)
scatter.to_excel("./xgb_train.xlsx", index=False)
