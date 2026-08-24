# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
       import pandas as pd 
       from scipy import stats 
       import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv") 
df.head()
```
<img width="298" height="217" alt="Screenshot 2026-08-24 074807" src="https://github.com/user-attachments/assets/b5f0c6aa-48ef-421f-8c33-c820349a9d0c" />

```
df_null_sum=df.isnull().sum() 
df_null_sum
```
<img width="96" height="223" alt="Screenshot 2026-08-24 074820" src="https://github.com/user-attachments/assets/33182cd4-0e89-416d-aa27-e07bf14a5df4" />

```
df.dropna()
```
<img width="316" height="448" alt="Screenshot 2026-08-24 074829" src="https://github.com/user-attachments/assets/fc9b2e8f-9f73-4a3b-9e5a-6422976328aa" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0) 
max_vals
```
<img width="172" height="173" alt="Screenshot 2026-08-24 081514" src="https://github.com/user-attachments/assets/06f37d7a-083f-4985-83cc-c131a81548b1" />

```
from sklearn.preprocessing import StandardScaler 
df1=pd.read_csv("/content/bmi.csv") 
df1.head()
```
<img width="316" height="448" alt="Screenshot 2026-08-24 074829" src="https://github.com/user-attachments/assets/8c30171d-71ab-452a-918e-0942698b94cc" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']]) 
df1.head(10)
```
<img width="324" height="384" alt="Screenshot 2026-08-24 074855" src="https://github.com/user-attachments/assets/33a51034-7e6a-4c60-bdbb-470f1b38dc8c" />

```
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler() 
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df.head(10)
```
<img width="320" height="389" alt="Screenshot 2026-08-24 074906" src="https://github.com/user-attachments/assets/8ffd2bca-876c-488f-b41b-379e0966a99d" />

```
from sklearn.preprocessing import MaxAbsScaler 
scaler = MaxAbsScaler() 
df3=pd.read_csv("/content/bmi.csv") 
df3.head()
```
<img width="291" height="215" alt="Screenshot 2026-08-24 074914" src="https://github.com/user-attachments/assets/f56535ae-21db-4100-bfcb-8fbe3a24eb03" />

```
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df
```
<img width="342" height="458" alt="Screenshot 2026-08-24 074922" src="https://github.com/user-attachments/assets/62d3b0b8-caf2-47ae-b15e-200955014655" />

```
from sklearn.preprocessing import RobustScaler 
scaler = RobustScaler() 
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']]) 
df3.head()
```
<img width="329" height="215" alt="Screenshot 2026-08-24 074937" src="https://github.com/user-attachments/assets/95ce0b70-4400-419e-885a-276cf30f5838" />

```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="396" height="399" alt="Screenshot 2026-08-24 074947" src="https://github.com/user-attachments/assets/0a50a1c1-8c80-4f35-8534-486c0d386122" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="169" height="545" alt="Screenshot 2026-08-24 074958" src="https://github.com/user-attachments/assets/dadcd22e-32b3-4f69-800a-d9141ecabe55" />

```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
# process non-numeric features. It ensures the model treats these columns as categories,
# not as continuous numerical values.
df[categorical_columns]
```
<img width="973" height="452" alt="Screenshot 2026-08-24 075015" src="https://github.com/user-attachments/assets/b550d9e5-07bb-46ae-ab56-c6a53f0a3cbf" />

```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```
<img width="813" height="455" alt="Screenshot 2026-08-24 075025" src="https://github.com/user-attachments/assets/fe6ad0ed-3847-47e7-89c6-52c993c3c470" />

```
df.dropna(inplace=True)
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict.

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="354" height="75" alt="Screenshot 2026-08-24 075044" src="https://github.com/user-attachments/assets/49ac4f27-31bf-4a98-a752-e8aae94d14c2" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
<img width="391" height="398" alt="Screenshot 2026-08-24 075051" src="https://github.com/user-attachments/assets/3fadc5db-115d-4740-9854-92d754ba432f" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="920" height="453" alt="Screenshot 2026-08-24 075100" src="https://github.com/user-attachments/assets/8d665435-0f8a-4426-92c4-76b7d7295f16" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="817" height="457" alt="Screenshot 2026-08-24 075110" src="https://github.com/user-attachments/assets/7b029b6a-6677-4e3b-9386-3cc9bf53d6c9" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
<img width="672" height="89" alt="Screenshot 2026-08-24 075119" src="https://github.com/user-attachments/assets/f0866081-0de5-4aab-9938-709e061b50fe" />

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
<img width="361" height="71" alt="Screenshot 2026-08-24 075128" src="https://github.com/user-attachments/assets/2042fd64-b9a9-4e1a-b9a7-526ccc9687db" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="512" height="36" alt="Screenshot 2026-08-24 075137" src="https://github.com/user-attachments/assets/2020d6d7-1d45-47ef-927e-22f42aa5c2af" />

```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
<img width="818" height="452" alt="Screenshot 2026-08-24 075157" src="https://github.com/user-attachments/assets/6be282bd-410b-418b-b4eb-a9c9f4aec7e5" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']

k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```
```
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
<img width="809" height="58" alt="Screenshot 2026-08-24 075206" src="https://github.com/user-attachments/assets/a70a63ae-0c51-485b-9a44-2ff0be5dfcfc" />

```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="810" height="461" alt="Screenshot 2026-08-24 075215" src="https://github.com/user-attachments/assets/1d6ac24b-0137-4c1d-9347-b8bcc64b4782" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
```
```
n_features_to_select = 5  # Example: Selecting 5 features
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
<img width="1017" height="799" alt="Screenshot 2026-08-24 075249" src="https://github.com/user-attachments/assets/4d140c76-c0c2-4ef3-a531-fab54202e03d" />

# RESULT:
    Thus, the Feature Scaling and Feature Selection processes were successfully performed on the given dataset, and the relevant features were selected for model building.
