import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib import font_manager, rc
from scipy import stats

raw_df = pd.read_csv("../bigfile/data_week4.csv", encoding='cp949')
df = raw_df.copy()

plt.rcParams.update({'font.family' : 'Malgun Gothic'}) # 맑은 고딕 설정
plt.rcParams.update({'axes.unicode_minus' : False}) # 음수 기호 깨짐 방지

# 자료형 변경
df['mold_code'] = df['mold_code'].astype('object')
df['registration_time'] = pd.to_datetime(df['registration_time'])
df['passorfail'] = df['passorfail'].astype('bool')

# 타겟변수에 있는 결측치 1개 제거하기
df.dropna(subset=['passorfail'], inplace=True)

# 불필요한 컬럼 제거
df = df.drop(['Unnamed: 0', 'line', 'name', 'mold_name', 'emergency_stop', 'time', 'date'], axis=1)

df2 = df2[(df2['physical_strength'] < 60000)&(df2['low_section_speed'] < 60000) & (df2['lower_mold_temp3'] < 60000)]
# ------------------------------------------------------------
## EDA (about 주조압력)
# 주조 압력
df['cast_pressure'].value_counts()

# 주조 압력 분포 분석
plt.figure(figsize=(10,6))
sns.histplot(df['cast_pressure'], bins=30, kde=True)
plt.title("주조 압력 분포")
plt.show()

# 주조 압력과 제품 품질 간의 관계 분석
plt.figure(figsize=(10, 6))
sns.boxplot(x='passorfail', y='cast_pressure', data=df)
plt.title("주조 압력 vs Pass/Fail")
plt.show()

# 주조 압력에 따른 불량률 변화
# 주조 압력을 구간별로 나누어 불량률 계산
# pressure_bin을 기준으로 불량 갯수 계산
df2['pressure_bin'] = pd.cut(df['cast_pressure'], bins=10)
fail_count = df[df['passorfail'] == True].groupby(['pressure_bin', 'mold_code'])['passorfail'].count().unstack()

# stacked bar plot으로 시각화
fail_count.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Fail Count by Cast Pressure and Mold Code")
plt.xlabel("Cast Pressure Range")
plt.ylabel("Fail Count")
plt.show()


# 시간에 따른 주조 압력의 변화 분석
df['date'] = df['registration_time'].dt.date
plt.figure(figsize=(10,6))
df.groupby('date')['cast_pressure'].mean().plot(kind='line', label='Average Cast Pressure')
df.groupby('date')['passorfail'].mean().plot(kind='line', secondary_y=True, label='Fail Rate', color='r')
plt.title("시간에 따른 주조 압력의 변화 분석")
plt.show()

# ------------------------------------------
## cast_pressure와 upper/lower_mold_temp 변수 상관관계
plt.figure(figsize=(10,6))
sns.scatterplot(x='cast_pressure', y='upper_mold_temp1', hue='passorfail', data=df)
plt.title("Cast Pressure vs Upper Mold Temperature 1")
plt.show()

## mold code별로 cast_pressure에 따른 불량률 변화 확인 (주조 압력에 따른 제품별 불량률 분석)
df['pressure_bin'] = pd.cut(df['cast_pressure'], bins=10)
mold_fail_rate = df.groupby(['pressure_bin', 'mold_code'])['passorfail'].mean().unstack()

mold_fail_rate.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Fail Rate by Cast Pressure and Mold Code")
plt.xlabel("Cast Pressure Range")
plt.ylabel("Fail Rate")
plt.show()

## 주조 압력과 production_cycletime 상관관계
plt.figure(figsize=(10,6))
sns.scatterplot(x='cast_pressure', y='production_cycletime', hue='passorfail', data=df)
plt.title("Cast Pressure vs Production Cycle Time")
plt.show()

correlation = df[['cast_pressure', 'production_cycletime']].corr()
print(correlation)

## 주조 압력이 불량률을 최소화하는 최적의 범위 도출해보기
df['pressure_bin'] = pd.cut(df['cast_pressure'], bins=np.arange(280, 350, 5))  # 주조 압력 세분화
optimal_fail_rate = df.groupby('pressure_bin')['passorfail'].mean()

plt.figure(figsize=(10,6))
optimal_fail_rate.plot(kind='bar')
plt.title("Optimal Fail Rate by Cast Pressure Range")
plt.xlabel("Cast Pressure Range")
plt.ylabel("Fail Rate")
plt.show()

## cast_pressure이 주기에 따라 어떻게 변화는지
df['month'] = df['registration_time'].dt.month
plt.figure(figsize=(10,6))
df.groupby('month')['cast_pressure'].mean().plot(kind='line', label='Average Cast Pressure')
df.groupby('month')['passorfail'].mean().plot(kind='line', secondary_y=True, label='Fail Rate', color='r')
plt.title("월별 주조 압력 및 불량률 분석")
plt.show()