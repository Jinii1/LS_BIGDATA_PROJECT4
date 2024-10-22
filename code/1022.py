# 전처리 X (기본) 베이스라인 성능 (sampling X, model: catboost)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지

# 데이터 불러오기
raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

df['tryshot_signal'].value_counts()

# 종속 변수와 독립 변수 설정
X = df.drop(columns=['passorfail'])
y = df['passorfail']

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)

# 데이터셋 분할 (훈련셋 80%, 테스트셋 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# CatBoost 모델
catboost_model = CatBoostClassifier(verbose=0, random_state=42)
catboost_model.fit(X_train, y_train)
y_pred_catboost = catboost_model.predict(X_test)
catboost_report = classification_report(y_test, y_pred_catboost)
print("CatBoost 모델 성능 보고서:\n", catboost_report)

# CatBoost 모델 성능 보고서:
#                precision    recall  f1-score   support

#        False       1.00      1.00      1.00     17600
#         True       0.98      0.93      0.96       803

#     accuracy                           1.00     18403
#    macro avg       0.99      0.97      0.98     18403
# weighted avg       1.00      1.00      1.00     18403
# --------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지


# 데이터 불러오기
raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)


# 종속 변수와 독립 변수 설정
X = df.drop(columns=['passorfail'])
y = df['passorfail']

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)

# 데이터셋 분할 (훈련셋 80%, 테스트셋 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# CatBoost 모델 학습
catboost_model = CatBoostClassifier(verbose=0, random_state=42)
catboost_model.fit(X_train, y_train)

# 예측
y_pred_catboost = catboost_model.predict(X_test)

# 성능 지표 구하기
catboost_report = classification_report(y_test, y_pred_catboost)

print("CatBoost 모델 성능 보고서:\n")
print(classification_report(y_test, y_pred_catboost, target_names=['False', 'True']))

# CatBoost 모델 성능 보고서:

#               precision    recall  f1-score   support

#        False       1.00      1.00      1.00     17600
#         True       0.98      0.93      0.96       803

#     accuracy                           1.00     18403
#    macro avg       0.99      0.97      0.98     18403
# weighted avg       1.00      1.00      1.00     18403
# ---------------------------------------------------------------








# 전처리 X (기본) 베이스라인 성능 (sampling X, model: lightgbm)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지

# 데이터 불러오기
raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

# 'registration_time'을 연도, 월, 일, 시간 등으로 분리
df['year'] = df['registration_time'].dt.year
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day
df['hour'] = df['registration_time'].dt.hour

# 원래의 'registration_time' 컬럼 제거
df = df.drop(columns=['registration_time'])

# 종속 변수와 독립 변수 설정
X = df.drop(columns=['passorfail'])
y = df['passorfail']

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)

# 데이터셋 분할 (훈련셋 80%, 테스트셋 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM 모델
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)
y_pred_lgbm = lgbm_model.predict(X_test)
lgbm_report = classification_report(y_test, y_pred_lgbm)
print("LightGBM 모델 성능 보고서:\n", lgbm_report)

# LightGBM 모델 성능 보고서:
#                precision    recall  f1-score   support

#        False       1.00      1.00      1.00     17600
#         True       0.98      0.93      0.95       803

#     accuracy                           1.00     18403
#    macro avg       0.99      0.97      0.98     18403
# weighted avg       1.00      1.00      1.00     18403

# ------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지


# 데이터 불러오기
raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
# lightgbm은 bool 형식 못받아들인다고 해서 코드 뻄 (-> 0 1로 진행)

# 시간 변수 추가
# lightgbm은 datetime 못알아듣는다고 해서 나눔
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date', 'registration_time'], axis=1)

# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# 종속 변수와 독립 변수 설정
X = df2.drop(columns=['passorfail'])
y = df2['passorfail']

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)

# 데이터셋 분할 (훈련셋 80%, 테스트셋 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# LightGBM 모델 학습
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train, y_train)

# 예측
y_pred_lgbm = lgbm_model.predict(X_test)

# 성능 지표 구하기
lgbm_report = classification_report(y_test, y_pred_lgbm)
print("LightGBM 모델 성능 보고서:\n", lgbm_report)

# LightGBM 모델 성능 보고서:
#                precision    recall  f1-score   support

#          0.0       1.00      1.00      1.00     17600
#          1.0       0.96      0.92      0.94       419

#     accuracy                           1.00     18019
#    macro avg       0.98      0.96      0.97     18019
# weighted avg       1.00      1.00      1.00     18019
# ------------------------------------------------
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지


# 데이터 불러오기
raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
# lightgbm은 bool 형식 못받아들인다고 해서 코드 뻄 (-> 0 1로 진행)

# 시간 변수 추가
# lightgbm은 datetime 못알아듣는다고 해서 나눔
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date', 'registration_time'], axis=1)

# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# 종속 변수와 독립 변수 설정
X = df2.drop(columns=['passorfail'])
y = df2['passorfail']

X.isna().sum()

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)
# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)
# 필요한 라이브러리 불러오기
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 종속 변수와 독립 변수 설정
X = df2.drop(columns=['passorfail'])
y = df2['passorfail']

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)

# 샘플링 기법 정의
samplers = {
    'RandomUnderSampler': RandomUnderSampler(random_state=42),
    'RandomOverSampler': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42)
}

results = {}

# 각 샘플링 기법 적용 후 CatBoost 모델 성능 평가
for sampler_name, sampler in samplers.items():
    # 샘플링 적용
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # 데이터셋 분할 (훈련셋 80%, 테스트셋 20%)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    # CatBoost 모델 학습
    catboost_model = CatBoostClassifier(verbose=0, random_state=42)
    catboost_model.fit(X_train, y_train)
    
    # 예측
    y_pred_catboost = catboost_model.predict(X_test)
    
    # 성능 지표 저장
    report = classification_report(y_test, y_pred_catboost, output_dict=True)
    results[sampler_name] = report

# 결과 출력
import pandas as pd
results_df = pd.DataFrame(results).T

# ----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지


# 데이터 불러오기
raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
# lightgbm은 bool 형식 못받아들인다고 해서 코드 뻄 (-> 0 1로 진행)

# 시간 변수 추가
# lightgbm은 datetime 못알아듣는다고 해서 나눔
df['hour'] = df['registration_time'].dt.hour
df['minute'] = df['registration_time'].dt.minute
df['second'] = df['registration_time'].dt.second
df['month'] = df['registration_time'].dt.month
df['day'] = df['registration_time'].dt.day

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date', 'registration_time'], axis=1)

# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# 종속 변수와 독립 변수 설정
X = df2.drop(columns=['passorfail'])
y = df2['passorfail']

# 범주형 변수를 one-hot encoding
X = pd.get_dummies(X)
# ---------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

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
# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

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

# 혼동행렬 구하기
conf_matrix = confusion_matrix(y_test, y_pred_catboost)

# 혼동행렬 시각화
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for CatBoost Model')
plt.show()

# 성능 지표 통합 형식으로 출력
precision = (conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall = (conf_matrix[1, 1]) / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)
support = conf_matrix.sum()

# 성능 지표 통합 결과 출력
print("\nCatBoost 모델 성능 보고서 (통합):")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")
print(f"Support: {support}")

# ----------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

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
# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

# tryshot_signal == nan 일 때만, df2로 만들기
df2 = df[df['tryshot_signal'].isna()]
df2 = df2.drop('tryshot_signal', axis=1)

# --------------------------------------------- 이상치 제거
df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000) & (df2['lower_mold_temp3'] < 60000)]
df2 = df2[df2['upper_mold_temp1']<1000]  # 고민해보기
df2 = df2[df2['upper_mold_temp2']<4000]  # 고민해보기

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

# Re-importing necessary packages
from sklearn.metrics import classification_report, confusion_matrix

# Generating the confusion matrix, recall, and precision values
cm = confusion_matrix(y_test, y_pred_catboost)
report = classification_report(y_test, y_pred_catboost, output_dict=True)

# Extracting recall and precision
precision = report['True']['precision']
recall = report['True']['recall']