import flask as Flask
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_excel("Sample data.xlsx",sheet_name="Sheet5")
ind_feat=df.columns
#print(ind_feat[0])
#print(df["NPS"].value_counts())
# replacing NA value in Geo to North America
df["Geo"]=df["Geo"].fillna("NA")
#print(df["Geo"].valuee_counts())
# #done with Geo
#updating data and remove notes in each column
df.iloc[:,3:]=df.iloc[:,3:].replace(to_replace =["0 - Not At All<br>Satisfied","0 - Not at<br>all Likely"],value ="0")
df.iloc[:,3:]= df.iloc[:,3:].replace(to_replace =["10 - Extremely<br>Satisfied","10 - Extremely<br>Likely"],value ="10")
"""unique_items=[]
for feature in ind_feat[3:]:
    a=df[feature].unique()
    print(a)
    if a in unique_items:
        pass
    else:
        unique_items.append(a)
print(unique_items)
l1=["Features_SAT_Value","Features_SAT_Design","Features_SAT_Weight","Features_SAT_Performance","Features_SAT_Quality","Equipment_SAT_OS","Equipment_SAT_Keyboard","Equipment_SAT_Noise","Equipment_SAT_Battery","NPS"]
unique_values=df[l1].nunique()
print(unique_values)"""
#print(df["Features_SAT_Value"].describe())
df["miss_more"]=df.isnull().sum(axis=1)
l1=df["miss_more"].unique()
sum_drop=0
list_drop=[]
list_rows_null=[]
for i in range(len(df)):
    if df["miss_more"][i]>0:
        list_rows_null.append(i)
    else:
        pass
print(len(list_rows_null))
for i in range(len(df)):
    if df["miss_more"][i]>5:
        df=df.drop(i)
        list_drop.append(i)
        sum_drop=sum_drop+1
    else:
        pass
print(len(list_drop))
df=df.drop(columns=['miss_more','Date'])#dropped added column & date column
"""print(sum)
print(list)
print(df.shape)
print(df.head())"""
#handling missing values
print(df.isnull().sum())
"""for feature in df.columns:
    data=df.copy()
    data.groupby(feature)['NPS'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('NPS')
    plt.title(feature)
    plt.show()"""
"""for feature in df.columns:
    data1 = df.copy()
    data1[feature] = np.where(data1[feature].isnull(), 1, 0)
    data1.groupby(feature)['NPS'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('NPS')
    plt.title(feature)
    plt.show()"""
#changing cateogorical features to numerical features
df1=pd.get_dummies(df["Geo"],drop_first=True)
df2=pd.get_dummies(df["LP"],drop_first=True)
df=pd.concat([df1,df2,df],axis=1)
df=df.drop(columns=['Geo','LP'])
print(df.columns)
l1_str=["Features_SAT_Value","Features_SAT_Design","Features_SAT_Weight","Features_SAT_Performance","Features_SAT_Quality","Equipment_SAT_OS","Equipment_SAT_Keyboard","Equipment_SAT_Noise","Equipment_SAT_Battery","NPS"]
for i in range(len(l1_str)):
    df[l1_str[i]]=df[l1_str[i]].astype(float)
df["Features_SAT_Value"] = df["Features_SAT_Value"].fillna(df["Features_SAT_Value"].mode()[0])
df["Features_SAT_Design"] = df["Features_SAT_Design"].fillna(df["Features_SAT_Design"].mode()[0])
df["Features_SAT_Weight"] = df["Features_SAT_Weight"].fillna(df["Features_SAT_Weight"].mode()[0])
df["Features_SAT_Quality"] = df["Features_SAT_Quality"].fillna(df["Features_SAT_Quality"].mode()[0])
df["Equipment_SAT_OS"] = df["Equipment_SAT_OS"].fillna(df["Equipment_SAT_OS"].mode()[0])
df["Features_SAT_Performance"] = df["Features_SAT_Performance"].fillna(df["Features_SAT_Performance"].mode()[0])
df["Equipment_SAT_Keyboard"] = df["Equipment_SAT_Keyboard"].fillna(df["Equipment_SAT_Keyboard"].mode()[0])
df["Equipment_SAT_Noise"] = df["Equipment_SAT_Noise"].fillna(df["Equipment_SAT_Noise"].mode()[0])
df["Equipment_SAT_Battery"] = df["Equipment_SAT_Battery"].fillna(df["Equipment_SAT_Battery"].mode()[0])
print(df.isnull().sum())
#X & Y assigning
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
#train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=32)
import xgboost
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
regressor=xgboost.XGBRegressor()
booster=['gbtree']
base_score=[0.25,0.5,0.75,1]
n_estimators = [100,300, 500,700, 900, 1100, 1500]
max_depth = [2, 3,4, 5,6,7,8, 10,12, 15]
learning_rate=[0.01,0.02,0.03,0.04,0.07,0.05,0.1,0.08,0.09]
min_child_weight=[0.5,1,2,3,4,5,0.3,0.2,0.7]
gamma=[0.1,0.2,0.3,0.5,0.7,0.8,1]
colsample_bytree=[0.2,0.3,0.5,0.7,1,1.2,1.5]
# Define the grid of hyperparameters to search
"""hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score,'colsample_bytree':colsample_bytree,'gamma':gamma }
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=10, n_iter=500,
            scoring = 'neg_mean_squared_error',n_jobs = 4,
            verbose = 5,
            return_train_score = True,
            random_state=42)
random_cv.fit(X_train,Y_train)
print(random_cv.best_estimator_)"""
regressor=xgboost.XGBRegressor(base_score=1, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.3, enable_categorical=False,
             gamma=0.1, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.01, max_delta_step=0,
             max_depth=4, min_child_weight=3,
             monotone_constraints='()', n_estimators=1100, n_jobs=12,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
regressor.fit(X_train,Y_train)
Y_pred=np.round(regressor.predict(X_test),0)
print(Y_pred)
from sklearn.metrics import r2_score
accuracy_train1=r2_score(Y_train,np.round(regressor.predict(X_train),0))
print(accuracy_train1)
accuracy=r2_score(Y_test,Y_pred)
print(accuracy)
import pickle
pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
from sklearn.model_selection import KFold
kfold_validation=KFold(5)
from sklearn.model_selection import cross_val_score
results=cross_val_score(regressor,X,Y,cv=kfold_validation)
print(results)
print(np.mean(results))