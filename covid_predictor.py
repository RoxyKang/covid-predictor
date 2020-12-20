# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import datetime
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

START_DATE = datetime.date(2019,12,31)
END_DATE = datetime.date(2020,10,25)

class linear_autoregressive():
    
    # k: number of days per time series line
    # f: number of the most relavent features to be selected
    def __init__(self, filename, k, f):
        self.filename = filename
        self.k = k
        self.features = f
    
    def fit(self, X, y):     
        self.w = solve(X.T@X, X.T@y)
    
    def predict(self, X):
        return X@self.w
    
    
    ############################ HELPER FUNCTIONS ############################
        
    def load_data(self):
        with open(os.path.join('/kaggle/input','cpsc-340-midterm-competition-phase2',self.filename), 'rb') as f:
            df = pd.read_csv(f, na_filter=False)
        
        df = df[df['cases_14_100k']>=0] # There are some garbage negative numbers
        self.X = df
        self.y = df
        print("[Script] dataset loaded.")
        
        
    def prep_data(self):
        self.load_data()
                
        df_X = self.X
                
        # Remove countries with too big/small deaths count
        df_X = self.select_country(df_X)
        
        # Get CA dataframe 
        df_CA = df_X[df_X['country_id']=='CA']
        
        df_CA_deaths = df_CA['deaths'] # Will be used as y
        
        df_CA = df_CA.drop('country_id',axis = 1)
        df_CA_datetime = self.datetime_reformat(df_CA) # Convert the dataframe that uses dates as indices
        df_CA_datetime.columns = ['casesCA', 'cases_100kCA', 'cases_14_100kCA', 'deathsCA'] # Rename column names
        
        # y
        pred_date = START_DATE + datetime.timedelta(days=self.k-1)
        df_y = df_CA_datetime[pred_date:]['deathsCA'].values.flatten()
        
        # Remove CA data for feature selection
        df_X = df_X.drop(df_X[df_X['country_id'] == 'CA'].index)
        
        # Feature selection
        df_X = df_X.drop('country_id',axis = 1) # Don't need to include country_id in feature selection
        df_X_datetime = self.datetime_reformat(df_X) # Convert the dataframe that uses dates as indices
        df_X = self.select_feature(df_X_datetime, df_CA_deaths, self.features) # Now df_X only contains selected features
        
        # Concat selected feature columns with CA columns
        df_X = pd.concat([df_CA_datetime, df_X],axis=1)
        
        # For each selected features, fit with an auto regressive model and filter by testError
        df = df_X
        feature_names = {}
        feature_w = {}
        
        feature_names, feature_w = self.select_features_by_ARTestError(df, feature_names, feature_w, 30, pred_date)
        feature_names, feature_w = self.select_features_by_ARTestError(df_CA_datetime, feature_names, feature_w, 400, pred_date)
              
        # Aggregate selected features into a final regressor for prediction
        final_regressor = pd.DataFrame.from_dict(feature_names)
        final_regressor_timeSeries = self.prep_timeSeries_pd(final_regressor, self.k)
        
        self.X = final_regressor_timeSeries
        self.y = df_y
        self.feature_names = feature_names
        self.feature_w = feature_w
                
        NX, DX = self.X.shape
        NY = self.y.shape[0]
        assert NX == NY
        
        print("[Script] X and y training dataset formatted.")
        
        
    def datetime_reformat(self, df_X):
        df_X = df_X.assign(cid = df_X.groupby(['date']).cumcount()).set_index(['date', 'cid']).unstack(-1).sort_index(1,1)
        df_X.columns = [f'{x}{y}' for x,y in df_X.columns]
        df_X.index = pd.to_datetime(df_X.index)
        df_X = df_X.sort_index()
        return df_X
            
    def prep_timeSeries_pd(self, df_X, k):
        # Make a new timeseries dataframe
        pred_start_date = START_DATE + datetime.timedelta(days=k-1)
        pred_end_date = END_DATE
        delta = pred_end_date - pred_start_date
        date_range = delta.days + 1
                
        df_val = df_X.values
        try:
            N, D = df_X.shape
        except:
            N = df_X.shape[0]
            D = 1
        
        ts_arr = np.empty(shape=(date_range,D*(k-1)+1))
        for i in range(date_range):
            ts_data = df_val[i:k+i-1]
            ts_data = ts_data.flatten()
            
            assert len(ts_data) == D*(k-1)
            
            # Append 1 in the front of each row
            ts_arr[i] = np.insert(ts_data, 0, 1)
        
        assert len(ts_arr) == (300-k+1)
        
        return ts_arr
    
    ########################## feature selection ##########################
    
    def select_features_by_ARTestError(self, df, feature_names, feature_w, errThreshold, pred_date):
        for i in range(len(df.columns)):
            X = df[df.columns[i]]
            df_i = self.prep_timeSeries_pd(X, self.k)
            y_i = X[pred_date:].values.flatten()
            w = solve(df_i.T@df_i, df_i.T@y_i)
            y_pred = df_i@w
            testError = np.mean((y_i - y_pred)**2)
            if(testError<errThreshold):
                y_pred = np.concatenate((X[:pred_date][:-1],y_pred), axis=0)
                feature_names[df.columns[i]] = y_pred
                feature_w[df.columns[i]] = w
        
        return feature_names, feature_w
    
    # Remove countries that have too small cumulative deaths
    def select_country(self, df):
        df1 = df.groupby('country_id')['deaths'].max().sort_values().iloc[160:].index.values
        df = df[df['country_id'].isin(df1)].reset_index(drop=True)
        return df
    
    def select_feature(self, df, df_CA_deaths, d):
        feature_index = self.forward_selection(df, df_CA_deaths, d).astype('int')
        column_index_names = df.columns.values[feature_index]
        df = df[column_index_names]
        return df
    
    def forward_selection(self,df, df_CA_deaths, num_feature):
        Count = 0
        df_X = df.values
        df_y = df_CA_deaths.values
        N, D = df_X.shape
        
        # set up an empty set of features
        S = np.array([])
        features = np.arange(D)
        
        X = np.zeros((N, num_feature))
        
        while len(S)<num_feature:
            best_score = np.inf
            best_feature = None
            rest_features = np.delete(features,S)
            
            for f in rest_features:
                X[:,len(S)] = df_X[:,f]
                try:
                    w = self.LeastSquares(X[:,:len(S)+1],df_y)
                    current_score = self.Score(w,X[:,:len(S)+1],df_y,'BIC')
                except:
                    Count = Count +1
                    continue
                if(current_score < best_score):
                    best_score = current_score
                    best_feature = f
                    
            X[:,len(S)] = df_X[:,best_feature]
            S = np.append(S,best_feature)
        return S
    
    def LeastSquares(self, X, y):
        bias = np.ones((len(X),1))
        Z = np.concatenate((bias, X), axis=1)
        v = solve(Z.T@Z, Z.T@y)
        return v
    
    def Score(self, w, X, df_y, criteria):
        score = 0
        if criteria == 'BIC':
            N,D = X.shape
            bias = np.ones((len(X),1))
            Z = np.concatenate((bias, X), axis=1)
            score = 1/2*(np.linalg.norm(np.dot(Z,w)-df_y))**2 + 1/2*np.log(N)*D
        return score
    
if __name__ == "__main__":
    start=datetime.datetime.now()
            
#     phase1_start_date = datetime.date(2020,10,6)
#     phase1_end_date = datetime.date(2020,10,16)
#     phase1_delta = phase1_end_date - phase1_start_date
#     phase1_date_range = phase1_delta.days + 1

    phase2_start_date = datetime.date(2020,10,26)
    phase2_end_date = datetime.date(2020,10,30)
    phase2_delta = phase2_end_date - phase2_start_date
    phase2_date_range = phase2_delta.days + 1
    
    model = linear_autoregressive('phase2_training_data.csv', 9, 25)
    model.prep_data()

    model.fit(model.X, model.y)
    yhat = model.predict(model.X)

    trainError = mean_squared_error(model.y, yhat)
    print("Training error = %.1f" % trainError)  

    features = model.feature_names
    for i in range(phase2_date_range):
        # 1. calculate features' prediction values
        for key in features:
            w = model.feature_w[key]
            X = np.insert(features[key][-model.k+1:], 0, 1)
            y_pred = np.dot(X.T,w)
            features[key] = np.append(features[key], y_pred)

        # 2. calculate CA death prediction values
        X = np.insert(yhat[-model.k+1:], 0, 1)
        y_pred = np.dot(X.T,w)
        yhat = np.append(yhat, y_pred)
    
    # Get the prediction of the last d days
    print(yhat[-6:])
    
    print("Elapsed time: ", datetime.datetime.now()-start)