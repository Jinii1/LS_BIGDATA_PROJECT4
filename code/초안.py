import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats
plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지

raw_df = pd.read_csv("bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

df.head()
df.info()
df.head()
df.tail()
df.columns
df.nunique()

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])


# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'time', 'date'], axis=1)
# ------------------------------------------------------------
# 주조 압력
df['cast_pressure'].value_counts()

# 주조 압력 분포 분석
plt.figure(figsize=(10,6))
sns.histplot(df['cast_pressure'], bins=30, kde=True)
plt.title("주조 압력 분포")
plt.show()

# 주조 압력과 제품 품질 간의 관계 분석
plt.figure(figsize=(10,6))
sns.boxplot(x='passorfail', y='cast_pressure', data=df)
plt.title("주조 압력 vs Pass/Fail")
plt.show()

# 주조 압력에 따른 불량률 변화
# 주조 압력을 구간별로 나누어 불량률 계산
df['pressure_bin'] = pd.cut(df['cast_pressure'], bins=10)
fail_rate = df.groupby('pressure_bin')['passorfail'].mean()

df['cast_pressure'].describe()

# 시각화
plt.figure(figsize=(10,6))
fail_rate.plot(kind='bar')
plt.title("Fail Rate by Cast Pressure Range")
plt.xlabel("Cast Pressure Range")
plt.ylabel("Fail Rate")
plt.show()

# 시간에 따른 주조 압력의 변화 분석
df['date'] = df['registration_time'].dt.date
plt.figure(figsize=(10,6))
df.groupby('date')['cast_pressure'].mean().plot(kind='line', label='Average Cast Pressure')
df.groupby('date')['passorfail'].mean().plot(kind='line', secondary_y=True, label='Fail Rate', color='r')
plt.title("시간에 따른 주조 압력의 변화 분석")
plt.show()

df['count']

# ------------------------------------------------------------




# 타겟 변수(passorfail) 분포 확인
df['passorfail'].value_counts().plot(kind='bar')
plt.title('Distribution of Pass/Fail')
plt.show()

# 온도 변수와 타겟 변수의 관계 분석
sns.boxplot(x='passorfail', y='upper_mold_temp1', data=df)
plt.title('Upper Mold Temperature 1 vs Pass/Fail')
plt.show()

# 시간에 따른 트렌드 분석
df['date'] = df['registration_time'].dt.date
daily_fail_rate = df.groupby('date')['passorfail'].mean()
daily_fail_rate.plot(kind='line')
plt.title('Daily Fail Rate Over Time')
plt.show()

# 컬럼별 고유값 개수 확인
for col in df.columns:
    print(f"{col}컬럼의 unique 개수 : {df[col].nunique()}")

# 숫자형 컬럼에 대한 히스토그램
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.show()

# 개별 변수를 기준으로 다른 변수들과 상관관계 분석
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 타겟 변수에 따른 상자그림(Box Plot)
plt.figure(figsize=(12, 6))
sns.boxplot(x='passorfail', y='upper_mold_temp1', data=df)
plt.title("Upper Mold Temperature 1 vs Pass/Fail")
plt.show()

# Cycle Time과 Pass/Fail 관계 KDE Plot
plt.figure(figsize=(12, 6))
sns.kdeplot(df[df['passorfail'] == 0]['facility_operation_cycleTime'], label='Pass')
sns.kdeplot(df[df['passorfail'] == 1]['facility_operation_cycleTime'], label='Fail')
plt.title("Cycle Time Distribution by Pass/Fail")
plt.show()

# 시간이 지나면서 불량률의 추세를 확인하는 라인 차트
df['date'] = df['registration_time'].dt.date
daily_fail_rate = df.groupby('date')['passorfail'].mean()
daily_fail_rate.plot(kind='line', figsize=(10, 6))
plt.title("Daily Fail Rate Over Time")
plt.show()

# mold_code 별로 등록된 시간의 범위 확인
df.groupby('mold_code')['registration_time'].agg(['min', 'max', 'count'])

df.info()
