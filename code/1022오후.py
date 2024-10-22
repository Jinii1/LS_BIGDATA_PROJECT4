## 이상치 제거: X, Sampling: X, Model: Catboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지

# 데이터 불러오기
raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 시간 변수 추가
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)


# -------------------------------------------- 새로운 데이터 프레임.
# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# --------------------------------------------- 이상치 제거
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000) & (df2['lower_mold_temp3'] < 60000)]
df2 = df2[df2['upper_mold_temp1']<1000]  # 고민해보기
df2 = df2[df2['upper_mold_temp2']<4000]  # 고민해보기

# ------------------------------------------- 파생변수
df2['molten_temp_g'] = np.where(df2['molten_temp']<600, 1,0)
df2['cast_pressure_g'] = np.where(df2['cast_pressure'] <= 270, 1, 0)
df2['biscuit_thickness_g'] = np.where((df2['biscuit_thickness']>60) |(df2['biscuit_thickness']<20), 1,0 )
df2['physical_strength_g'] = np.where(df2['physical_strength'] < 500, 1, 0)
df2['low_section_speed_g'] = np.where(df2['low_section_speed'] < 75, 1, 0)
df2['high_section_speed_g'] = np.where((df2['high_section_speed'] >200)|(df2['high_section_speed'] < 70), 1, 0)
# ---------------------------------------------------------------------------------------------------





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from catboost import CatBoostClassifier
from imblearn.metrics import geometric_mean_score

# 종속 변수와 독립 변수 설정
X = df2.drop(columns=['passorfail'])
y = df2['passorfail']

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)

# 데이터셋 분할 (훈련셋 80%, 테스트셋 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# CatBoost 모델 학습
catboost_model = CatBoostClassifier(verbose=0, random_state=42)
catboost_model.fit(X_train, y_train)

# 예측
y_pred_catboost = catboost_model.predict(X_test)
y_pred_proba = catboost_model.predict_proba(X_test)[:, 1]  # ROC-AUC에 사용할 확률 예측

# 혼동행렬 시각화
cm = confusion_matrix(y_test, y_pred_catboost)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for CatBoost Model')
plt.show()

# 성능 지표 계산
report = classification_report(y_test, y_pred_catboost, output_dict=True)
precision = report['True']['precision']
recall = report['True']['recall']
f1 = f1_score(y_test, y_pred_catboost)
roc_auc = roc_auc_score(y_test, y_pred_proba)
g_mean = geometric_mean_score(y_test, y_pred_catboost)

# 결과 출력
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"G-Mean: {g_mean:.4f}")
# ----------------------------------------------------------------------------------------






##  변수 중요도 확인
# 변수 중요도 계산
feature_importances = catboost_model.get_feature_importance()
features = X.columns

# 변수 중요도를 데이터프레임으로 정리
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# 중요도 높은 순으로 정렬
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 전체 변수 중요도 출력
print("전체 변수 중요도:")
print(feature_importance_df)

# 변수 중요도 시각화 (Top 10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Top 10 변수 중요도 출력
print("\nTop 10 변수 중요도:")
print(feature_importance_df.head(10))
# ----------------------------------------------------------------------------------------































import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지

# 데이터 불러오기
raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])

# 시간 변수 추가
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)


# -------------------------------------------- 새로운 데이터 프레임.
# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# --------------------------------------------- 이상치 제거
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000) & (df2['lower_mold_temp3'] < 60000)]
df2 = df2[df2['upper_mold_temp1']<1000]  # 고민해보기
df2 = df2[df2['upper_mold_temp2']<4000]  # 고민해보기

# ------------------------------------------- 파생변수
df2['molten_temp_g'] = np.where(df2['molten_temp']<600, 1,0)
df2['cast_pressure_g'] = np.where(df2['cast_pressure'] <= 270, 1, 0)
df2['biscuit_thickness_g'] = np.where((df2['biscuit_thickness']>60) |(df2['biscuit_thickness']<20), 1,0 )
df2['physical_strength_g'] = np.where(df2['physical_strength'] < 500, 1, 0)
df2['low_section_speed_g'] = np.where(df2['low_section_speed'] < 75, 1, 0)
df2['high_section_speed_g'] = np.where((df2['high_section_speed'] >200)|(df2['high_section_speed'] < 70), 1, 0)


# -------------------------------------------- 모델 돌리기 전 과정
# 종속 변수와 독립 변수 설정
X = df2.drop(columns=['passorfail'])
y = df2['passorfail']

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)

# 데이터셋 분할 (훈련셋 80%, 테스트셋 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# -------------------------------------------- 이상치 파생변수

# 이상치 여부 컬럼 만들기
def IQR_outlier(data) :
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR) 
    upper_bound = Q3 + (1.5 * IQR)
    out_df = pd.concat([lower_bound, upper_bound], axis = 1).T
    out_df.index = ['하한','상한']
    return out_df

num_X_train = X_train.select_dtypes(include=('number'))

for col in num_X_train.columns:
	X_train[f'{col}_outlier'] = np.where((X_train[col]<IQR_outlier(num_X_train).loc['하한',col])|(X_train[col]>IQR_outlier(num_X_train).loc['상한',col]),True,False)
	X_test[f'{col}_outlier'] = np.where((X_test[col]<IQR_outlier(num_X_train).loc['하한',col])|(X_test[col]>IQR_outlier(num_X_train).loc['상한',col]),True,False)
      
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from catboost import CatBoostClassifier
from imblearn.metrics import geometric_mean_score

# CatBoost 모델 학습
catboost_model = CatBoostClassifier(verbose=0, random_state=42)
catboost_model.fit(X_train, y_train)

# 예측
y_pred_catboost = catboost_model.predict(X_test)
y_pred_proba = catboost_model.predict_proba(X_test)[:, 1]  # ROC-AUC에 사용할 확률 예측

# 혼동행렬 시각화
cm = confusion_matrix(y_test, y_pred_catboost)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for CatBoost Model')
plt.show()

from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.metrics import geometric_mean_score

# 성능 지표 계산
report = classification_report(y_test, y_pred_catboost, output_dict=True)

# 1 클래스에 해당하는 precision, recall 계산
precision = report[1]['precision']
recall = report[1]['recall']

# f1-score, roc-auc, g-means 계산
f1 = f1_score(y_test, y_pred_catboost)
roc_auc = roc_auc_score(y_test, y_pred_proba)
g_mean = geometric_mean_score(y_test, y_pred_catboost)

# 결과 출력
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"G-Mean: {g_mean:.4f}")



##  변수 중요도 확인
# 변수 중요도 계산
feature_importances = catboost_model.get_feature_importance()
features = X.columns

# 변수 중요도를 데이터프레임으로 정리
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# 중요도 높은 순으로 정렬
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 전체 변수 중요도 출력
print("전체 변수 중요도:")
print(feature_importance_df)

# 변수 중요도 시각화 (Top 10)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Top 10 변수 중요도 출력
print("\nTop 10 변수 중요도:")
print(feature_importance_df.head(10))