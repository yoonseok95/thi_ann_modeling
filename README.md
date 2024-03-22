# THI_ANN_modeling

### libraries
import pandas as pd  
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import StandarScaler  
from sklearn.metrics import mealn_absolute_error  
from sklearn.metrics import mean_squared_error  
from sklearn.metrics import r2_score  
import statsmedels.api as sm  
from statsmedels.stats.outliers_influence import variance_inflation_factor  
from sklearn.model_selection import KFold  
from sklearn.model_selection import cross_val_score  
from sklear.model_selection import train_test_split  
from keras.models import Sequential  
from keras.layers import Dense  
inport warnings  
warnings.filterwarnings("ignore)  
plt.rc("font", family = "Malgun Gothic")  
sns.set(font="Malgon Gothic", rc={"axes.unicodde_minus":False}, style='white')   

### data
df = pd.read_excel("C:/Users/MOON/Desktop/2024 바이오소재공학연구실/모델식 데이터.xlsx")  
DataFrame = df.drop(columns = ['소구분'], axis = 1)  
DataFrame.head()

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/669fddd1-4601-4288-82fb-f272d888b452)  
> 결과로 table을 얻을 수 있다.
- - -  

# 기술통계량 확인(df)
des_df = round(DataFrame.descriev(), 4)  
des_df  

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/59610206-62bc-492c-bde2-1ef5f736dc57)  
> 결과로 table을 얻을 수 있다.
- - -  

# 이상치 제거    
DataFrame_1 = DataFrame[['피부온도']].dropna(axis = 0)  
print(DataFrame_1)  
def remove_outliers_iqr(data, column_name, threshold=):  
    Q1 = np.percentile(data[column_name], 25) # 1사분위수  
    Q3 = np.percentile(data[column_name], 75) # 3사분위수  
    IQR = Q3 - Q1 # IQR 계산  

#*이상치를 탐지하여 제거하는 과정*    
    lower_bound = Q1 - threshold * IQR  
    upper_bound = Q3 + threshold * IQR   
    outlier_indices = data[(data[column_name] < lower_bound) | (data[column_name] ` upper_bound)].index  
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]  

return data, outlier_indices   

Data, Outlier_indices = remove_outliers_iqr(DataFram_1, '피부온도')  
print(Data.describe())  
print(Outlier_indices)  


> 결과로 피부 온도를 확인할 수 있다.  
> count  44.000000  
> mean   31.206818  
> std     0.925956  
> min    30.000000  
> 25%    30.400000  
> 50%    31.050000  
> 75%    31.750000  
> max    33.400000  
> Int64Index([], dtype='int64')  
- - -  

### DataFram에서 종속 변수 열 선택  
dependent_variable = DataFram['유량']  

#*이상치 제거 기준 설정 (예: Z-score를 사용한 방법)*  
z_scores = np.abs((dependent_variable - dependent_variable.mean()) / dependent_variable.std())  
z_scores = np.abs((dependent_variable - dependent_variable.mean()) / dependent_variable.std())  
threshold = 3 *임계값 설정*  
print(z_scores)  

#*이상치의 인덱스 확인*  
outlier_indices = z_scores[z_scores > threshold].index   
print("이상치의 인덱스:")  
print(outlier_indices)  

#*이상치 제거*  
data_de = DataFram[z_scores <= threshold]  
round(data_de.describe(), 4)  


> #input file  
> 0      2.709091  
> 1      1.935295    
> 2      2.978854  
> 3      2.432228  
> 4      2.336391  
>          ...     
> 359    0.478379  
> 360    0.002743  
> 361    0.057599  
> 362    0.771052  
> 363    0.117941  
> 
> Name: 유량, Length: 364, dtype: float64  
> 이상치의 인덱스: Int64Index([8, 25], dtype='int64')  
> 
> #output file
> 결과로 table을 얻을 수 있다.
![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/54a7b236-d567-41af-bc4c-1ed41021d1db)  

- - -  

### 이상치 제거 데이터  

df_1 = pd.read_excel("C:/Users/MOON/Desktop/2024 바이오소재공학연구실/모델식 데이터.xlsx")  
DataFram_ou = df_1.drop(columns = ['소구분'], axis = 1)  
#*기술통계량 확인*  
des_df_ou = round(DataFram_ou.describe(), 4)  
des_df_ou

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/47563400-dfb5-4b87-8bed-ae9b87729d15)  
> 결과로 table을 얻을 수 있다.  
- - -  
 
### Data normalization  
nor_df = (DataFrame_ou - DataFram_ou.min()) / (DataFrame_ou.max() - DataFram)ou.min())  
print(nor_df.describe())  
#*Undo normalization for y value*  
y_unnormalized = nor_df['유량'] * (DataFrame_ou['유량'].max() - DataFram_ou['유량'].min()) + DataFram_ou['유량'].min()  
#*Remove flow column from nor_df*  
nor_df.drop('유량', axis = 1, inplce = True)  
#*Add y_unnormalized column to nor_df*  
nor_df['유량'] = y_unnormalized  
print(nor_of.describe())

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/368f8d83-5fc0-4088-ae2b-5975f778de62)  
> 결과로 table을 얻을 수 있다.

- - -  

### 변수간 Correlation 확인 (결측치 제거 이후)  
plt.figure(figsize = (14,10))  
sns.heatmap(nor_df.corr(), annot = True)


![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/078a580c-3bd4-49d6-aafb-fa7c70387785)  
> 결과로 이런 이미지를 확인할 수 있다.  
- - -  

### 변수 구분  
x = nor_df.drop(columns = ['유량','온도'], axis =1)  
x = x.dropna(axis = 0).reset_index(drop=True)  
y = nor_df[['유량']]  
#*독립변수 행렬 생성*  
matrix_x = np.array(x)  
#*VIF 점수 계산*  
vif = pd.DataFrame()  
vif['Features'] = x.columns  
vif['VIF Score'] = [variance_inflation_factor(matrix_x,i) for i in range(matrix_x.shape[1])]  

print(vif) 

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/0439e7fa-b0a9-48d6-bea4-022d17ebd508)  


> 결과로 table을 얻을 수 있다.
- - -  

### 변수 구분  
x = nor_df.drop(columns = ['유량','온도','피부온도','음수섭취량'], axis =1)  
x = x.dropna(axis=0).reset_index(drop=True)  
y = nor_df[['유량']]  
#*독립변수 행렬 생성*   
matrix_x = np.array(x)  
#*VIF 점수 계산*  
vif = pd.DataFrame()  
vif['Features'] = x.columns  
vif['VIF Score'] = [variance_inflation_factor(matrix_x,i) for i in range(matrix_x.shape[1])]  

print(vif)  

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/e70abb88-7e52-4999-a1c0-97fd51bb03df)  

> 결과로 table을 얻을 수 있다.
- - -  

### 결측치 제거 (DIM, 직장온도 포함)  
df_na = nor_df.dropna(axis=0).reset_index(drop=True)  
print(df_na.describe())  
x = df_na.drop(columns = ['유량', '온도', '피부온도', '음수섭취량'], axis = 1)  
y = df_na[['유량']]


![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/0593b604-802c-447e-be28-8b313feef7a1)  

> 결과로 table을 얻을 수 있다.
- - - 

### 결측치 제거 (직장온도 포함,변수 3개)  
df_drop = nor_df.drop(columns = ['DIM', '음수섭취량', '온도','피부온도'])  
print(df_drop.head())  
df_na_1 = df_drop.dropna(axis=0).reset_index(drop=True)  
x_1 = df_na_1.drop(columns = ['유량'], axis = 1)  
y_1 = df_na_1[['유량']]  
print(df_na_1.describe())  

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/19c7904f-7b37-4360-9fe1-8761374098bf)  

> 결과로 table을 얻을 수 있다.  
- - -  

### 결측치 제거 (직장온도 포함, 변수 2개)
x_2 = df_na_1.drop(columns = ['유량', '사료섭취량'])  
y_2 = df_na_1[['유량']]  
print(x_2.describe())

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/10274329-1234-418c-bbcf-829f39eddce2)  

> 결과로 table을 얻을 수 있다.  
- - -  

### 결측치 제거 (직장온도 제외, 변수 2개)  
df_drop_1 = nor_df.drop(columns = ['DIM', '음수섭취량', '온도', '피부온도','직장온도'])  
print(df_drop_1.head())  
df_na_2 = df_drop_1.dropna(axis=0).reset_index(drop=True)  
x_3 = df_na_2.drop(columns = ['유량'], axis = 1)  
y_3 = df_na_2[['유량']]  
print(df_na_2.describe())

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/a8c6bb9b-3265-48b7-a421-d0aa94d7b8c8)  

> 결과로 table을 얻을 수 있다. 
- - -  

### 결측치 제거 (THI)   
df_drop_2 = nor_df[['유량','THI']]  
df_na_3 = df_drop_2.dropna(axis=0).reset_index(drop=True)  
x_4 = df_na_3[['THI']]  
y_4 = df_na_3[['유량']]  
print(df_na_3.describe())  

![image](https://github.com/yoonseok95/thi_ann_modeling/assets/145320578/04c9d2e9-861c-4cea-9511-b7a22256a1c5)  
   
> 결과로 table을 얻을 수 있다.  
- - -  
### 결측치 제거 (DIM, 직장온도 포함, 변수 2개)  
df_na = nor_df.dropna(axis=0).reset_index(drop=True)  
print(df_na.describe())  
x_5 = df_na.drop(columns = ['유량', '온도', '피부온도', '음수섭취량','사료섭취량'], axis = 1)  
y_5 = df_na[['유량']]  
- - -  
### 결측치 제거 (DIM만 포함)  
df_drop = nor_df.drop(columns = ['온도', '피부온도', '음수섭취량','사료섭취량','직장온도'], axis = 1)  
print(df_drop.head())  
df_na_1 = df_drop.dropna(axis=0).reset_index(drop=True)  
print(df_na_1.describe())  
x_6 = df_na_1.drop(columns = ['유량'], axis = 1)  
y_6 = df_na_1[['유량']]  
print(y_1)  
- - -  
### Data split  
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x, y, test_size = 0.2, random_state = 42)  

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_1, y_1, test_size = 0.2, random_state = 42)  

x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(x_2, y_2, test_size = 0.2, random_state = 42)  

x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(x_3, y_3, test_size = 0.2, random_state = 42)  

x_train_5, x_test_5, y_train_5, y_test_5 = train_test_split(x_4, y_4, test_size = 0.2, random_state = 42)  

#*training data set의 인덱스 초기화*  
x_train_1 = x_train_1.reset_index(drop=True)  
y_train_1 = y_train_1.reset_index(drop=True)  

x_train_2 = x_train_2.reset_index(drop=True)  
y_train_2 = y_train_2.reset_index(drop=True)  

x_train_3 = x_train_3.reset_index(drop=True)  
y_train_3 = y_train_3.reset_index(drop=True)  

x_train_4 = x_train_4.reset_index(drop=True)  
y_train_4 = y_train_4.reset_index(drop=True)  

x_train_5 = x_train_5.reset_index(drop=True)  
y_train_5 = y_train_5.reset_index(drop=True)  
- - -  
### MLR(machine learning x)  
def MLR(x, y):  
    model = LinearRegression()  
    model.fit(x, y)  
    
#*Get coefficients and intercept*  
coefficients = model.coef_  
intercept = model.intercept_  
    
MAPE = np.mean(100 * (np.abs(y-model.predict(x))/y))  
accuracy = 100 - MAPE  
#*Calculating RMSE*  
y_pred = model.predict(x)  
rmse = np.sqrt(np.mean((y - y_pred)**2))  
#*Calculating Relative RMSE*  
relative_rmse = (rmse / np.mean(y))*100  
#*Calculating R-squared*  
r2 = r2_score(y, y_pred)  
#*Printing the results of the current fold iteration*  
print('Coefficient:', coefficients)  
print('Intercept:', intercept)  
print('Accuracy:', accuracy, 'RMSE', rmse, 'RRMSE', relative_rmse, 'r2', r2)  
- - -  
MLR_1 = MLR(x,y)  
print("----------------------------------------------------------------------")  
MLR_2 = MLR(x_1,y_1)  
print("----------------------------------------------------------------------")  
MLR_3 = MLR(x_2,y_2)  
print("----------------------------------------------------------------------")  
MLR_4 = MLR(x_3,y_3)  
print("----------------------------------------------------------------------")  
MLR_5 = MLR(x_4,y_4)  
print("----------------------------------------------------------------------")  
MLR_6 = MLR(x_5,y_5)  
print("----------------------------------------------------------------------")  
MLR_7 = MLR(x_6,y_6)  
- - -  
def Multiple(x_train, y_train, x_test, y_test, k_fold=10):  
    SearchResultsData=pd.DataFrame()  
    #*Create MLR model*  
    model = LinearRegression()  
     #*Perform k-fold cross validation*  
    kf = KFold(n_splits=k_fold, shuffle=True, random_state = 5)  
    fold_number = 1  
    for train_index, val_index in kf.split(x_train):  
        X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]  
        Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]  
        
 #*Fitting MLR to the Training set*  
 model.fit(X_train_fold, Y_train_fold)  
        
#*Get coefficients and intercept*  
coefficients = model.coef_  
intercept = model.intercept_  

MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))  
        MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))  
        accuracy_val = 100 - MAPE_val  
        accuracy = 100 - MAPE  
        #*Calculating RMSE*  
        y_pred_val = model.predict(X_val_fold)  
        y_pred = model.predict(x_test)  
        rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))  
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))  
        #*Calculating Relative RMSE*  
        relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100  
        relative_rmse = (rmse / np.mean(y_test))*100  
        #*Calculating R-squared*  
        r2_val = r2_score(Y_val_fold, y_pred_val)  
        r2_test = r2_score(y_test, y_pred)  
        #*Printing the results of the current fold iteration*  
        print('Fold:', fold_number)  
        print('Coefficient:', coefficients)  
        print('Intercept:', intercept)  
        print('Accuracy_val:', accuracy_val,'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val',  
              relative_rmse_val, 'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test)  
        fold_number += 1  
        #*Appending the results to the dataframe*  
        SearchResultsData = pd.concat([SearchResultsData,  
                                       pd.DataFrame(data=[[fold_number,coefficients, intercept, accuracy_val, accuracy, rmse_val,  
                                                           relative_rmse_val, rmse, relative_rmse]],  
                                                    columns=['Fold', 'Coefficients', 'intercept', 'Accuracy_val', 'Accuracy_test', 'RMSE_val',  
                                                             'RRMSE_val', 'RMSE_test', 'RRMSE_test'])])  
    
 return(SearchResultsData)  
- - -  
### Calling the function   
MLR_1 = Multiple(x_train_1, y_train_1, x_test_1, y_test_1, k_fold=10)  
MLR_1.to_excel('MLR1.xlsx', index=False)  
#Calling the function  
MLR_2 = Multiple(x_train_2, y_train_2, x_test_2, y_test_2, k_fold=10)  
MLR_2.to_excel('MLR2.xlsx', index=False)  
#Calling the function  
MLR_3 = Multiple(x_train_3, y_train_3, x_test_3, y_test_3, k_fold=10)  
MLR_3.to_excel('MLR3.xlsx', index=False)  
#Calling the function  
MLR_4 = Multiple(x_train_4, y_train_4, x_test_4, y_test_4, k_fold=10)  
MLR_4.to_excel('MLR4.xlsx', index=False)  
#Calling the function  
MLR_5 = Multiple(x_train_5, y_train_5, x_test_5, y_test_5, k_fold=10)  
MLR_5.to_excel('MLR5.xlsx', index=False)  

> 결과로 다수의 fold를 결과로 확인할 수 있다.  
- - -  
### import kerastuner as kt  
 
def build_model(hp):  
    model = Sequential()  
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),   
                    activation='relu', input_shape=(x_train_1.shape[1],)))  
    for i in range(hp.Int('num_layers', 1, 5)):  
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),  
                        activation='relu'))  
    model.add(Dense(1, activation='linear'))  
    model.compile(optimizer='adam', loss='mean_squared_error')  
    return model  
- - -  
tuner = kt.RandomSearch(  
    build_model,  
    objective='val_loss',  
    max_trials=10,  
    executions_per_trial=5,  
    directory='my_dir_3',  
    project_name='ANN_final_model_1')  

tuner.search(x_train_1, y_train_1, epochs=10, validation_data=(x_test_1, y_test_1))    

#*Get the optimal hyperparameters*  
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]  

print(f"""  
The hyperparameter search is complete. The optimal number of units in the first densely-connected  
layer is {best_hps.get('units')} and the optimal number of layers is {best_hps.get('num_layers')}.  
""")

> 결과로   
>  Reloading Tuner from my_dir_3\ANN_final_model_1\tuner0.json  
>    
>  The hyperparameter search is complete. The optimal number of units in the first densely-connected  
>  layer is 480 and the optimal number of layers is 5.    
> 
> 확인할 수 있습니다.  
- - -  
 
### layer 별 노드개수(model1)  
print("layer 1:", best_hps.get('units_0'), "layer 2", best_hps.get("units_1"), "layer 3", best_hps.get("units_2"),   
      "layer 4", best_hps.get("units_3"), "layer 5", best_hps.get("units_4"))

> 결과로 **layer 1: 320 layer 2 512 layer 3 256 layer 4 480 layer 5 32**를 얻었다.  
- - -  

### Defining a function to find the best parameters for ANN and obtain results for the training dataset  
def FunctionFindBestParams_1(x_train, y_train, x_test, y_test, k_fold=10):  
    
#*Defining the list of hyper parameters to try*  
batch_size_list=[5, 10, 15, 20, 25, 30, 35]  
epoch_list  =   [5, 10, 50, 100, 250, 500]  
    
SearchResultsData=pd.DataFrame()  
    
#*Initializing the trials*  
TrialNumber=0  
for batch_size_trial in batch_size_list:  
for epochs_trial in epoch_list:  
TrialNumber+=1  
model = tuner.hypermodel.build(best_hps)  

#*Perform k-fold cross validation*  
kf = KFold(n_splits=k_fold, shuffle=True, random_state = 5)  
fold_number = 1  
for train_index, val_index in kf.split(x_train):  
X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]  
Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]  
                
#*Fitting the ANN to the Training set*  
                model.fit(X_train_fold, Y_train_fold ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)  
                MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))  
                MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))  
                accuracy_val = 100 - MAPE_val  
                accuracy = 100 - MAPE  
                
#*Calculating RMSE*  
y_pred_val = model.predict(X_val_fold)  
y_pred = model.predict(x_test)  
rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))  
rmse = np.sqrt(np.mean((y_test - y_pred)**2))  

#*Calculating Relative RMSE*  
relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100  
relative_rmse = (rmse / np.mean(y_test))*100  
                
#*Calculating R-squared*  
r2_val = r2_score(Y_val_fold, y_pred_val)  
r2_test = r2_score(y_test, y_pred)  

#*Printing the results of the current fold iteration*  
print('Fold:', fold_number, 'TrialNumber:', TrialNumber, 'Parameters:', 'batch_size:',  
      batch_size_trial, '-', 'epochs:', epochs_trial, 'Accuracy_val:', accuracy_val,  
      'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val', relative_rmse_val,  
      'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test)  
                
 fold_number += 1  
            
            
#*Appending the results to the dataframe*  
SearchResultsData = pd.concat([SearchResultsData,  
pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial),   
                    accuracy_val, accuracy, rmse_val, relative_rmse_val, rmse,  
                    relative_rmse]],  
columns=['TrialNumber', 'Parameters', 'Accuracy_val', 'Accuracy_test',  
         'RMSE_val', 'RRMSE_val', 'RMSE_test', 'RRMSE_test'])])  
                
return(SearchResultsData)  

- - -  

### Calling the function  
ANN_1 = FunctionFindBestParams_1(x_train_1, y_train_1, x_test_1, y_test_1, k_fold=10)  
ANN_1.to_excel('ANN_Final_1.xlsx', index=False)  

> 결과로 다수의 fold 값을 확인할 수 있다.  
- - -  

def build_model_2(hp):  
    model = Sequential()  
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),   
                    activation='relu', input_shape=(x_train_2.shape[1],)))  
    for i in range(hp.Int('num_layers', 1, 5)):  
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),  
                        activation='relu'))  
    model.add(Dense(1, activation='linear'))  
    model.compile(optimizer='adam', loss='mean_squared_error')  
    return model  
- - -  

tuner2 = kt.RandomSearch(  
    build_model_2,  
    objective='val_loss',  
    max_trials=10,  
    executions_per_trial=5,  
    directory='my_dir_3',  
    project_name='ANN_final_model_2')  

tuner2.search(x_train_2, y_train_2, epochs=10, validation_data=(x_test_2, y_test_2))  

#Get the optimal hyperparameters  
best_hps_2 = tuner2.get_best_hyperparameters(num_trials=1)[0]  

print(f"""  
The hyperparameter search is complete. The optimal number of units in the first densely-connected  
layer is {best_hps_2.get('units')} and the optimal number of layers is {best_hps_2.get('num_layers')}.  
""")  

> 결과로   
> Reloading Tuner from my_dir_3\ANN_final_model_2\tuner0.json  
> 
> The hyperparameter search is complete. The optimal number of units in the first densely-connected  
> layer is 512 and the optimal number of layers is 4.  
> 확인할 수 있다.  
- - -  

### layer 별 노드개수(model 1)  
print("layer 1:", best_hps_2.get('units_0'), "layer 2", best_hps_2.get("units_1"), "layer 3", best_hps_2.get("units_2"),  
      "layer 4", best_hps_2.get("units_3")) 

> 결과로 **layer 1: 288 layer 2 160 layer 3 256 layer 4 416** 얻었다.  
- - -  

### Defining a function to find the best parameters for ANN and obtain results for the training dataset  

def FunctionFindBestParams_2(x_train, y_train, x_test, y_test, k_fold=10):  
    
#*Defining the list of hyper parameters to try*  
batch_size_list=[5, 10, 15, 20, 25, 30, 35]  
epoch_list  =   [5, 10, 50, 100, 250, 500]  
    
SearchResultsData=pd.DataFrame()  
    
#*Initializing the trials*  
TrialNumber=0  
for batch_size_trial in batch_size_list:  
        for epochs_trial in epoch_list:  
            TrialNumber+=1  
            model = tuner2.hypermodel.build(best_hps_2)  
            # Perform k-fold cross validation  
            kf = KFold(n_splits=k_fold, shuffle=True, random_state = 5)  
            fold_number = 1  
            for train_index, val_index in kf.split(x_train):  
                X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]  
                Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]  
                
#*Fitting the ANN to the Training set*  
                model.fit(X_train_fold, Y_train_fold ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)  
                MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))  
                MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))  
                accuracy_val = 100 - MAPE_val  
                accuracy = 100 - MAPE  
                
#*Calculating RMSE*  
                y_pred_val = model.predict(X_val_fold)  
                y_pred = model.predict(x_test)  
                rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))  
                rmse = np.sqrt(np.mean((y_test - y_pred)**2))  
                
#*Calculating Relative RMSE*  
                relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100  
                relative_rmse = (rmse / np.mean(y_test))*100  
                
#*Calculating R-squared*  
                r2_val = r2_score(Y_val_fold, y_pred_val)  
                r2_test = r2_score(y_test, y_pred)  
                
#*Printing the results of the current fold iteration*   
                print('Fold:', fold_number, 'TrialNumber:', TrialNumber, 'Parameters:', 'batch_size:',  
                      batch_size_trial, '-', 'epochs:', epochs_trial, 'Accuracy_val:', accuracy_val,  
                      'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val', relative_rmse_val,  
                      'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test)  
                
fold_number += 1  
            
            
#*Appending the results to the dataframe*   
                SearchResultsData = pd.concat([SearchResultsData,  
                                               pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial),  
                                                                   accuracy_val, accuracy, rmse_val, relative_rmse_val, rmse,  
                                                                   relative_rmse]],  
                                                            columns=['TrialNumber', 'Parameters', 'Accuracy_val', 'Accuracy_test',  
                                                                     'RMSE_val', 'RRMSE_val', 'RMSE_test', 'RRMSE_test'])])  
                
return(SearchResultsData)  
- - -  

### Calling the function   
ANN_2 = FunctionFindBestParams_2(x_train_2, y_train_2, x_test_2, y_test_2, k_fold=10)  
ANN_2.to_excel('ANN_Final_2.xlsx', index=False)  

> 결과로 다수의 fold 값을 확인할 수 있다.   
- - -  

def build_model_3(hp):  
    model = Sequential()  
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),   
                    activation='relu', input_shape=(x_train_3.shape[1],)))  
    for i in range(hp.Int('num_layers', 1, 5)):  
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),  
                        activation='relu'))  
    model.add(Dense(1, activation='linear'))  
    model.compile(optimizer='adam', loss='mean_squared_error')  
    return model  
- - -  

tuner3 = kt.RandomSearch(  
    build_model_3,  
    objective='val_loss',  
    max_trials=10,  
    executions_per_trial=5,  
    directory='my_dir_3',  
    project_name='ANN_final_model_3')  

tuner3.search(x_train_3, y_train_3, epochs=10, validation_data=(x_test_3, y_test_3))  

#*Get the optimal hyperparameters*  
best_hps_3 = tuner3.get_best_hyperparameters(num_trials=1)[0]  

print(f"""  
The hyperparameter search is complete. The optimal number of units in the first densely-connected  
layer is {best_hps_3.get('units')} and the optimal number of layers is {best_hps_3.get('num_layers')}.  
""")  
- - -  

### layer 별 노드개수(model1)   
print("layer 1:", best_hps_3.get('units_0'), "layer 2", best_hps_3.get("units_1"), "layer 3", best_hps_3.get("units_2"),  
      "layer 4", best_hps_3.get("units_3"))  

> 결과로 **layer 1: 384 layer 2 192 layer 3 416 layer 4 288**을 확인할 수 있다.  
- - -  

### Defining a function to find the best parameters for ANN and obtain results for the training dataset  
def FunctionFindBestParams_3(x_train, y_train, x_test, y_test, k_fold=10):  
    
#*Defining the list of hyper parameters to try*  
    batch_size_list=[5, 10, 15, 20, 25, 30, 35]  
    epoch_list  =   [5, 10, 50, 100, 250, 500]  
    
SearchResultsData=pd.DataFrame()  
     
#*Initializing the trials*  
    TrialNumber=0  
    for batch_size_trial in batch_size_list:  
        for epochs_trial in epoch_list:  
            TrialNumber+=1  
            model = tuner3.hypermodel.build(best_hps_3)  
            # Perform k-fold cross validation  
            kf = KFold(n_splits=k_fold, shuffle=True, random_state = 5)  
            fold_number = 1  
            for train_index, val_index in kf.split(x_train):  
                X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]  
                Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]  
                
#*Fitting the ANN to the Training set*  
                model.fit(X_train_fold, Y_train_fold ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)  
                MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))  
                MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))  
                accuracy_val = 100 - MAPE_val  
                accuracy = 100 - MAPE  
                
#*Calculating RMSE*  
                y_pred_val = model.predict(X_val_fold)  
                y_pred = model.predict(x_test)  
                rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))  
                rmse = np.sqrt(np.mean((y_test - y_pred)**2))  
               
#*Calculating Relative RMSE*  
                relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100  
                relative_rmse = (rmse / np.mean(y_test))*100  
                
#*Calculating R-squared*  
                r2_val = r2_score(Y_val_fold, y_pred_val)  
                r2_test = r2_score(y_test, y_pred)  
                
#*Printing the results of the current fold iteration*  
                print('Fold:', fold_number, 'TrialNumber:', TrialNumber, 'Parameters:', 'batch_size:',  
                      batch_size_trial, '-', 'epochs:', epochs_trial, 'Accuracy_val:', accuracy_val,  
                      'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val', relative_rmse_val,  
                      'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test)  
                
fold_number += 1  
            
            
#*Appending the results to the dataframe*  
                SearchResultsData = pd.concat([SearchResultsData,  
                                               pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial),  
                                                                   accuracy_val, accuracy, rmse_val, relative_rmse_val, rmse,  
                                                                   relative_rmse]],  
                                                            columns=['TrialNumber', 'Parameters', 'Accuracy_val', 'Accuracy_test',  
                                                                     'RMSE_val', 'RRMSE_val', 'RMSE_test', 'RRMSE_test'])])  
                
return(SearchResultsData)  
- - -  

### Calling the function  
ANN_3 = FunctionFindBestParams_3(x_train_3, y_train_3, x_test_3, y_test_3, k_fold=10)  
ANN_3.to_excel('ANN_Final_3.xlsx', index=False)  

> 결과로 다수의 fold를 확인할 수 있다.  
- - -  

def build_model_4(hp):  
    model = Sequential()  
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),   
                    activation='relu', input_shape=(x_train_4.shape[1],)))  
    for i in range(hp.Int('num_layers', 1, 5)):  
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),  
                        activation='relu'))  
    model.add(Dense(1, activation='linear'))  
    model.compile(optimizer='adam', loss='mean_squared_error')  
    return model  
    
- - -  
 
tuner4 = kt.RandomSearch(  
    build_model_4,  
    objective='val_loss',  
    max_trials=10,  
    executions_per_trial=5,  
    directory='my_dir_3',  
    project_name='ANN_final_model_4')  

tuner4.search(x_train_4, y_train_4, epochs=10, validation_data=(x_test_4, y_test_4))  

# Get the optimal hyperparameters    
best_hps_4 = tuner4.get_best_hyperparameters(num_trials=1)[0]  

print(f"""  
The hyperparameter search is complete. The optimal number of units in the first densely-connected  
layer is {best_hps_4.get('units')} and the optimal number of layers is {best_hps_4.get('num_layers')}.  
""")  

> 결과로  
> Trial 10 Complete [00h 00m 15s]  
> val_loss: 37.534515380859375  
>    
> Best val_loss So Far: 36.62616729736328  
> Total elapsed time: 00h 02m 40s  
>   
> The hyperparameter search is complete. The optimal number of units in the first densely-connected  
> layer is 96 and the optimal number of layers is 5.
> 
> 확인할 수 있다.  
- - -  
 
### layer 별 노드개수(model 1)    
print("layer 1:", best_hps_4.get('units_0'), "layer 2", best_hps_4.get("units_1"), "layer 3", best_hps_4.get("units_2"),  
      "layer 4", best_hps_4.get("units_3"), "layer 5", best_hps_4.get("units_4"))   

> 결과값으로 **layer 1: 64 layer 2 32 layer 3 480 layer 4 32 layer 5 32**를 얻었다.   
- - -  

### Defining a function to find the best parameters for ANN and obtain results for the training dataset  
def FunctionFindBestParams_4(x_train, y_train, x_test, y_test, k_fold=10):  
    
#*Defining the list of hyper parameters to try*  
    batch_size_list=[5, 10, 15, 20, 25, 30, 35]  
    epoch_list  =   [5, 10, 50, 100, 250, 500]  
    
SearchResultsData=pd.DataFrame()  
     
#*Initializing the trials*  
    TrialNumber=0  
    for batch_size_trial in batch_size_list:  
        for epochs_trial in epoch_list:  
            TrialNumber+=1  
            model = tuner4.hypermodel.build(best_hps_4)  
#*Perform k-fold cross validation*  
            kf = KFold(n_splits=k_fold, shuffle=True, random_state = 5)  
            fold_number = 1  
            for train_index, val_index in kf.split(x_train):  
                X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]  
                Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]  
                
#*Fitting the ANN to the Training set*  
                model.fit(X_train_fold, Y_train_fold ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)  
                MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))  
                MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))  
                accuracy_val = 100 - MAPE_val  
                accuracy = 100 - MAPE  
                
#*Calculating RMSE*  
                y_pred_val = model.predict(X_val_fold)  
                y_pred = model.predict(x_test)  
                rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))  
                rmse = np.sqrt(np.mean((y_test - y_pred)**2))  
                
#*Calculating Relative RMSE*  
                relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100  
                relative_rmse = (rmse / np.mean(y_test))*100  
                
#*Calculating R-squared*  
                r2_val = r2_score(Y_val_fold, y_pred_val)  
                r2_test = r2_score(y_test, y_pred)  
                
#*Printing the results of the current fold iteration*  
                print('Fold:', fold_number, 'TrialNumber:', TrialNumber, 'Parameters:', 'batch_size:',  
                      batch_size_trial, '-', 'epochs:', epochs_trial, 'Accuracy_val:', accuracy_val,  
                      'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val', relative_rmse_val,  
                      'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test)  
                
fold_number += 1  
            
            
#*Appending the results to the dataframe*  
                SearchResultsData = pd.concat([SearchResultsData,  
                                               pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial),  
                                                                   accuracy_val, accuracy, rmse_val, relative_rmse_val, rmse,  
                                                                   relative_rmse]],  
                                                            columns=['TrialNumber', 'Parameters', 'Accuracy_val', 'Accuracy_test',  
                                                                     'RMSE_val', 'RRMSE_val', 'RMSE_test', 'RRMSE_test'])])  
                
return(SearchResultsData)  
- - -  

### Calling the function   
ANN_4 = FunctionFindBestParams_4(x_train_4, y_train_4, x_test_4, y_test_4, k_fold=10)  
ANN_4.to_excel('ANN_Final_4.xlsx', index=False)  

> 결과로 다수의 fold 값을 얻었다.  
- - -  

def build_model_5(hp):  
    model = Sequential()  
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),   
                    activation='relu', input_shape=(x_train_5.shape[1],)))  
    for i in range(hp.Int('num_layers', 1, 5)):  
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),  
                        activation='relu'))  
    model.add(Dense(1, activation='linear'))  
    model.compile(optimizer='adam', loss='mean_squared_error')  
    return model  
- - -  

tuner5 = kt.RandomSearch(  
    build_model_5,  
    objective='val_loss',  
    max_trials=10,  
    executions_per_trial=5,  
    directory='my_dir_3',  
    project_name='ANN_final_model_5')  

tuner5.search(x_train_5, y_train_5, epochs=10, validation_data=(x_test_5, y_test_5))  

#*Get the optimal hyperparameters*  
best_hps_5 = tuner5.get_best_hyperparameters(num_trials=1)[0]  

print(f"""  
The hyperparameter search is complete. The optimal number of units in the first densely-connected  
layer is {best_hps_5.get('units')} and the optimal number of layers is {best_hps_5.get('num_layers')}.  
""")  

> 결과로  
> Reloading Tuner from my_dir_3\ANN_final_model_5\tuner0.json  
>   
> The hyperparameter search is complete. The optimal number of units in the first densely-connected  
> layer is 192 and the optimal number of layers is 4.
>   
> 확인할 수 있다.  
- - -  

### layer 별 노드개수(model 1)  
print("layer 1:", best_hps_5.get('units_0'), "layer 2", best_hps_5.get("units_1"), "layer 3", best_hps_5.get("units_2"),  
      "layer 4", best_hps_5.get("units_3"))  

> 결과로 **layer 1: 160 layer 2 128 layer 3 448 layer 4 416**을 얻었다.  
- - -  

### Defining a function to find the best parameters for ANN and obtain results for the training dataset  
def FunctionFindBestParams_5(x_train, y_train, x_test, y_test, k_fold=10):  
    
#*Defining the list of hyper parameters to try*  
    batch_size_list=[5, 10, 15, 20, 25, 30, 35]  
    epoch_list  =   [5, 10, 50, 100, 250, 500]  
    
SearchResultsData=pd.DataFrame()  
    
#*Initializing the trials*  
    TrialNumber=0  
    for batch_size_trial in batch_size_list:  
        for epochs_trial in epoch_list:  
            TrialNumber+=1  
            model = tuner5.hypermodel.build(best_hps_5)  
            
#*Perform k-fold cross validation*  
            kf = KFold(n_splits=k_fold, shuffle=True, random_state = 5)  
            fold_number = 1  
            for train_index, val_index in kf.split(x_train):  
                X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]  
                Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]  
                
#*Fitting the ANN to the Training set*  
                model.fit(X_train_fold, Y_train_fold ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)  
                MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))  
                MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))  
                accuracy_val = 100 - MAPE_val  
                accuracy = 100 - MAPE  
                
#*Calculating RMSE*  
                y_pred_val = model.predict(X_val_fold)  
                y_pred = model.predict(x_test)  
                rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))  
                rmse = np.sqrt(np.mean((y_test - y_pred)**2))  
                
#*Calculating Relative RMSE*  
                relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100  
                relative_rmse = (rmse / np.mean(y_test))*100  
                
#*Calculating R-squared*  
                r2_val = r2_score(Y_val_fold, y_pred_val)  
                r2_test = r2_score(y_test, y_pred)  
                
#*Printing the results of the current fold iteration*  
                print('Fold:', fold_number, 'TrialNumber:', TrialNumber, 'Parameters:', 'batch_size:',  
                      batch_size_trial, '-', 'epochs:', epochs_trial, 'Accuracy_val:', accuracy_val,  
                      'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val', relative_rmse_val,  
                      'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test)  
                
fold_number += 1  
            
            
#*Appending the results to the dataframe*    
                SearchResultsData = pd.concat([SearchResultsData,  
                                               pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial),  
                                                                   accuracy_val, accuracy, rmse_val, relative_rmse_val, rmse,  
                                                                   relative_rmse]],  
                                                            columns=['TrialNumber', 'Parameters', 'Accuracy_val', 'Accuracy_test',  
                                                                     'RMSE_val', 'RRMSE_val', 'RMSE_test', 'RRMSE_test'])])  
                
return(SearchResultsData)  
- - -  

### Calling the function   
ANN_5 = FunctionFindBestParams_5(x_train_5, y_train_5, x_test_5, y_test_5, k_fold=10)  
ANN_5.to_excel('ANN_Final_5.xlsx', index=False)   

> 결과로 다수의 fold 값을 확인할 수 있다.  
- - -  
def build_model_6(hp):  
    model = Sequential()  
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),   
                    activation='relu', input_shape=(x_train_6.shape[1],)))  
    for i in range(hp.Int('num_layers', 1, 5)):  
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),  
                        activation='relu'))  
    model.add(Dense(1, activation='linear'))  
    model.compile(optimizer='adam', loss='mean_squared_error')  
    return model  
- - -  
tuner_6 = kt.RandomSearch(  
    build_model_6,  
    objective='val_loss',  
    max_trials=10,  
    executions_per_trial=3,  
    directory='my_dir_6',  
    project_name='keras_tuner_regression_6')  

tuner_6.search(x_train_6, y_train_6, epochs=10, validation_data=(x_test_6, y_test_6))  

#*Get the optimal hyperparameters*  
best_hps_6 = tuner_6.get_best_hyperparameters(num_trials=1)[0]  

print(f"""  
The hyperparameter search is complete. The optimal number of units in the first densely-connected  
layer is {best_hps_6.get('units')} and the optimal number of layers is {best_hps_6.get('num_layers')}.  
""")  
print(best_hps_6.get('units_0'))  
print()  
- - -   
#layer 별 노드개수(model 2)  
print("layer 1:", best_hps_6.get('units_0'), "layer 2", best_hps_6.get("units_1"),  
      "layer 3", best_hps_6.get("units_2"),  "layer 4", best_hps_6.get("units_3"), "layer 5", best_hps_6.get("units_4") )  
- - -  
### Defining a function to find the best parameters for ANN and obtain results for the training dataset   
def FunctionFindBestParams_kt_6(x_train, y_train, x_test, y_test, k_fold=10):    
    
#*Defining the list of hyper parameters to try*  
batch_size_list=[5, 10, 15, 20, 25, 30, 35]  
epoch_list  =   [5, 10, 50, 100, 250, 500]  
    
SearchResultsData=pd.DataFrame()  
    
#*Initializing the trials*  
TrialNumber=0  
for batch_size_trial in batch_size_list:  
    for epochs_trial in epoch_list:  
        TrialNumber+=1  
        # best dense layer ANN model  
        model = tuner.hypermodel.build(best_hps)  
            
#*Perform k-fold cross validation*  
kf = KFold(n_splits=k_fold, shuffle=True, random_state = 42)  
fold_number = 1  
for train_index, val_index in kf.split(x_train):  
X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]  
Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]  
                
#*Fitting the ANN to the Training set*  
model.fit(X_train_fold, Y_train_fold ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)  
                
#*Calculating the coefficients and intercept of the linear regression model*  
coefficients = model.layers[-1].get_weights()[0]  
intercept = model.layers[-1].get_weights()[1]  
            
MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))  
MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))  
accuracy_val = 100 - MAPE_val  
accuracy = 100 - MAPE  

#*Calculating RMSE*  
y_pred_val = model.predict(X_val_fold)  
y_pred = model.predict(x_test)  
rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))  
rmse = np.sqrt(np.mean((y_test - y_pred)**2))  

#*Calculating Relative RMSE*  
relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100  
relative_rmse = (rmse / np.mean(y_test))*100  

#*Calculating R-squared*  
r2_val = r2_score(Y_val_fold, y_pred_val)  
r2_test = r2_score(y_test, y_pred)  

#*Printing the results of the current fold iteration*  
print('Fold:', fold_number, 'TrialNumber:', TrialNumber)  
print('Coefficients:', coefficients)  
print('Intercept:', intercept)  
print('Parameters:', 'batch_size:',  
        batch_size_trial, '-', 'epochs:', epochs_trial, 'Accuracy_val:', accuracy_val,  
        'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val', relative_rmse_val,  
        'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test)  
                
fold_number += 1  
            
            
#*Appending the results to the dataframe*  
SearchResultsData = pd.concat([SearchResultsData,  
                                               pd.DataFrame(data=[[TrialNumber,coefficients,intercept, str(batch_size_trial)+'-'+str(epochs_trial),  
                                                                   accuracy_val, accuracy, rmse_val, relative_rmse_val, rmse,  
                                                                   relative_rmse]],  
                                                            columns=['TrialNumber','Coefficients','Intercept', 'Parameters', 'Accuracy_val', 'Accuracy_test',  
                                                                     'RMSE_val', 'RRMSE_val', 'RMSE_test', 'RRMSE_test'])])  
                
return(SearchResultsData)    
- - -  
### Calling the function  
ANN_6 = FunctionFindBestParams_6(x_train_6, y_train_6, x_test_6, y_test_6, k_fold=10)  
ANN_6.to_excel('ANN_Final_6.xlsx', index=False)  
- - -  
def build_model_7(hp):  
    model = Sequential()  
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),   
                    activation='relu', input_shape=(x_train_7.shape[1],)))  
    for i in range(hp.Int('num_layers', 1, 5)):  
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),  
                        activation='relu'))  
    model.add(Dense(1, activation='linear'))  
    model.compile(optimizer='adam', loss='mean_squared_error')  
    return model  
- - -  
tuner_7 = kt.RandomSearch(  
    build_model_7,  
    objective='val_loss',  
    max_trials=10,  
    executions_per_trial=3,  
    directory='my_dir_7',  
    project_name='keras_tuner_regression_7')  

tuner_7.search(x_train_7, y_train_7, epochs=10, validation_data=(x_test_7, y_test_7))  

#*Get the optimal hyperparameters*  
best_hps_7 = tuner_7.get_best_hyperparameters(num_trials=1)[0]  

print(f"""  
The hyperparameter search is complete. The optimal number of units in the first densely-connected  
layer is {best_hps_7.get('units')} and the optimal number of layers is {best_hps_7.get('num_layers')}.  
""")  
print(best_hps_7.get('units_0'))  
print()  
- - -  
### layer 별 노드개수(model 2)  
print("layer 1:", best_hps_7.get('units_0'), "layer 2", best_hps_7.get("units_1"),  
      "layer 3", best_hps_7.get("units_2"),  "layer 4", best_hps_7.get("units_3"), "layer 5", best_hps_7.get("units_4") )    
- - -   
### Defining a function to find the best parameters for ANN and obtain results for the training dataset  
def FunctionFindBestParams_kt_7(x_train, y_train, x_test, y_test, k_fold=10):  
    
#*Defining the list of hyper parameters to try*  
batch_size_list=[5, 10, 15, 20, 25, 30, 35]  
epoch_list  =   [5, 10, 50, 100, 250, 500]  
    
SearchResultsData=pd.DataFrame()  
    
#*Initializing the trials*  
TrialNumber=0  
for batch_size_trial in batch_size_list:  
    for epochs_trial in epoch_list:  
        TrialNumber+=1  
        
#*best dense layer ANN model*  
model = tuner.hypermodel.build(best_hps)  
            
#*Perform k-fold cross validation*  
kf = KFold(n_splits=k_fold, shuffle=True, random_state = 42)  
fold_number = 1  
for train_index, val_index in kf.split(x_train):  
    X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]  
    Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]  
                
#*Fitting the ANN to the Training set*  
model.fit(X_train_fold, Y_train_fold ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)  
                
#*Calculating the coefficients and intercept of the linear regression model*  
coefficients = model.layers[-1].get_weights()[0]  
intercept = model.layers[-1].get_weights()[1]  
            
MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))  
MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))  
accuracy_val = 100 - MAPE_val  
accuracy = 100 - MAPE  

#*Calculating RMSE*  
y_pred_val = model.predict(X_val_fold)  
y_pred = model.predict(x_test)  
rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))  
rmse = np.sqrt(np.mean((y_test - y_pred)**2))  

#*Calculating Relative RMSE*  
relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100  
relative_rmse = (rmse / np.mean(y_test))*100  

#*Calculating R-squared*  
r2_val = r2_score(Y_val_fold, y_pred_val)  
r2_test = r2_score(y_test, y_pred)  

#*Printing the results of the current fold iteration*  
print('Fold:', fold_number, 'TrialNumber:', TrialNumber)  
print('Coefficients:', coefficients)  
print('Intercept:', intercept)  
print('Parameters:', 'batch_size:',  
        batch_size_trial, '-', 'epochs:', epochs_trial, 'Accuracy_val:', accuracy_val,  
        'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val', relative_rmse_val,  
        'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test)  
                
fold_number += 1  
            
#*Appending the results to the dataframe*  
SearchResultsData = pd.concat([SearchResultsData,  
                                               pd.DataFrame(data=[[TrialNumber,coefficients,intercept, str(batch_size_trial)+'-'+str(epochs_trial),  
                                                                   accuracy_val, accuracy, rmse_val, relative_rmse_val, rmse,  
                                                                   relative_rmse]],  
                                                            columns=['TrialNumber','Coefficients','Intercept', 'Parameters', 'Accuracy_val', 'Accuracy_test',  
                                                                     'RMSE_val', 'RRMSE_val', 'RMSE_test', 'RRMSE_test'])])  
                  
return(SearchResultsData)  
- - -  
### Calling the function  
ANN_7 = FunctionFindBestParams_7(x_train_7, y_train_7, x_test_7, y_test_7, k_fold=10)  
ANN_7.to_excel('ANN_Final_7.xlsx', index=False)  
- - -  
