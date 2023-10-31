# Laporan Proyek Machine Learning
### Nama : Reza Raditya
### Nim :  231352004
### Kelas : Pagi B

## Domain Proyek

Menganalisis data pasien diabetes   
**
## Business Understanding
Dalam dataset ini mencakup data data tentang klasifikasi level atau tingkat pasien terkena diabetes  
Bagian laporan ini mencakup:

### Problem Statements

- bagaimana identifikasi tingkat diabetes pasien


### Goals

-  mengklasifikasikan data dengan secara detail sesuai identifikasi



 ### Solution statements
    - dapat mengklasifikan data dan mencari data yang sesuai dengan yang diinputkan costumer

## Data Understanding
dataset : https://www.kaggle.com/code/ryotapy/diabetes-logistic-regression/notebook

### Variabel-variabel diabetes dadtaset                                                                                                 adalah sebagai berikut:
 0   Pregnancies              : data kehamilan        
 1   Glucose                  : data kadar gula             
 2   BloodPressure            : data tekanan darah           
 3   SkinThickness            : data ketebalan kulit             
 4   Insulin                  : data hormon        
 5   BMI                      : data berat badan  
 6   DiabetesPedigreeFunction : data riwayat keturunan diabetes 
 7   Age                      : data umur   
 8   Outcome                  : data hasil  
 ### variabel eda
 ![image](https://github.com/rezaraditya/data/assets/60649124/39a8b83f-f764-4c57-9c62-af091d55e9a7)
<br>
![image](https://github.com/rezaraditya/data/assets/60649124/75a87c0d-756f-44f6-be26-8c543aaee1bc)
<br>
![image](https://github.com/rezaraditya/data/assets/60649124/d39118eb-3137-4222-b7bd-0c4c9d903d70)
<br>
## Data Preparation
# Mengimport libary
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

# to display all columns
pd.set_option("display.max_columns",None) Memanggil Dataset
df = pd.read_csv('toyota.csv')
## Data Loading
TARGET = "Outcome"
df = pd.read_csv('diabetes.csv')
print(df.shape)
df.head(3)
(768, 9)
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
## Data detail
df.describe()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
count	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000	768.000000
mean	3.845052	120.894531	69.105469	20.536458	79.799479	31.992578	0.471876	33.240885	0.348958
std	3.369578	31.972618	19.355807	15.952218	115.244002	7.884160	0.331329	11.760232	0.476951
min	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.078000	21.000000	0.000000
25%	1.000000	99.000000	62.000000	0.000000	0.000000	27.300000	0.243750	24.000000	0.000000
50%	3.000000	117.000000	72.000000	23.000000	30.500000	32.000000	0.372500	29.000000	0.000000
75%	6.000000	140.250000	80.000000	32.000000	127.250000	36.600000	0.626250	41.000000	1.000000
max	17.000000	199.000000	122.000000	99.000000	846.000000	67.100000	2.420000	81.000000	1.000000
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
plt.figure(figsize=(15,25))
for i in range(len(df.columns)):
    plt.subplot(521+i)
    sns.histplot(df.iloc[:,i])
plt.show()



# fill with median
median_BloodPressure = df['BloodPressure'].median()
df['BloodPressure'] = df['BloodPressure'].replace(0, median_BloodPressure)

median_BMI = df[TARGET].median()
df['BMI'] = df[TARGET].replace(0, median_BMI)

plt.figure(figsize=(12,8))
plt.subplot(221)
sns.boxplot(x=df[TARGET],y=df["Glucose"])

plt.subplot(222)
sns.boxplot(x=df[TARGET],y=df["SkinThickness"])

plt.subplot(223)
sns.boxplot(x=df[TARGET],y=df["Insulin"])
plt.show()

tmp_df = df.copy()

median_Glucose = tmp_df['Glucose'].median()
median_SkinThickness = tmp_df['SkinThickness'].median()
median_Insulin = tmp_df['Insulin'].median()
tmp_df['Glucose'] = tmp_df['Glucose'].replace(0, median_Glucose)
tmp_df['SkinThickness'] = tmp_df['SkinThickness'].replace(0, median_SkinThickness)
tmp_df['Insulin'] = tmp_df['Insulin'].replace(0, median_Insulin)

plt.figure(figsize=(12,8))
plt.subplot(221)
sns.boxplot(x=tmp_df[TARGET],y=tmp_df["Glucose"])

plt.subplot(222)
sns.boxplot(x=tmp_df[TARGET],y=tmp_df["SkinThickness"])

plt.subplot(223)
sns.boxplot(x=tmp_df[TARGET],y=tmp_df["Insulin"])
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

Model
train_df = tmp_df.iloc[:700,:]
test_df = tmp_df.iloc[700:,:]
X_train = train_df.drop(TARGET,axis=1)
y_train = train_df[TARGET]
X_test = test_df.drop(TARGET,axis=1)
y_test = test_df[TARGET]
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
Model Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

coefficients = model.coef_[0]
intercept = model.intercept_[0]

coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': coefficients})
intercept_df = pd.DataFrame({'Feature': ['Intercept'], 'Coefficient': [intercept]})
coef_df = pd.concat([coef_df, intercept_df], ignore_index=True)

print("Coefficients:")
display(coef_df)
## Modeling
# membuat model regresi linier
-from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
pred = lr.predict(X_test)

- score = lr.score(X_test, y_test)
print('akurasi model regresi linear= ', score)


## save model
- import pickle
filename = 'diabet.sav'
pickle.dump(model,open(filename,'wb')
**Jelaskan proses improvement yang dilakukan**.
- Proses Improvement ini menggunakan  LogisticRegression

## Evaluation
- tidak ada evaluasi


## Deployment
https://dataset-hzeytbrghafkrmxgjfqt6n.streamlit.app/

**---Ini adalah bagian akhir laporan---**

_
