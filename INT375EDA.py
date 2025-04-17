import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.weightstats import ztest


df = pd.read_csv("space_missions_dataset.csv")
df['Launch Date'] = pd.to_datetime(df['Launch Date'])


print(df.head())
print(df.tail())

print("=== Null Values in Each Column ===")
print(df.isnull().sum())
df.dropna(inplace=True)

# === TABLES ===
print("\n=== Table 1: Descriptive Statistics ===")
print(df.describe())
print("\n",df.info())
print("\n=== Table 2: Mission Counts by Target Type ===")
print(df['Target Type'].value_counts())

print("\n=== Table 3: Avg. Mission Success by Mission Type ===")
print(df.groupby('Mission Type')['Mission Success (%)'].mean().round(2))

print("\n=== Table 4: Avg. Scientific Yield by Launch Vehicle ===")
print(df.groupby('Launch Vehicle')['Scientific Yield (points)'].mean().round(2))

print("\n=== Table 5: Total Fuel Consumption per Target ===")
print(df.groupby('Target Name')['Fuel Consumption (tons)'].sum().round(2).sort_values(ascending=False))

# === VISUALS ===
sns.set(style='whitegrid')

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df.select_dtypes(include=np.number))
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

# Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Mission Cost (billion USD)", y="Scientific Yield (points)", hue="Mission Type")
plt.title("Scatterplot: Mission Cost vs Scientific Yield")
plt.show()

# Histogram + KDE
plt.figure(figsize=(8, 4))
sns.histplot(df['Distance from Earth (light-years)'], kde=True, bins=15, color='skyblue')
plt.title("Histogram + KDE: Distance from Earth")
plt.show()

# Violinplot
plt.figure(figsize=(8, 4))
sns.violinplot(data=df, x='Mission Type', y='Scientific Yield (points)', hue='Mission Type', palette='pastel', legend=False)
plt.title("Violin Plot: Scientific Yield by Mission Type")
plt.show()

# Swarmplot
plt.figure(figsize=(10, 5))
sns.swarmplot(data=df, x='Launch Vehicle', y='Crew Size', hue='Mission Type')
plt.title("Swarmplot: Crew Size by Launch Vehicle")
plt.xticks(rotation=45)
plt.show()

# KDE Plot 
plt.figure(figsize=(8, 4))
sns.kdeplot(df['Fuel Consumption (tons)'], fill=True, color='darkorange')
plt.title("KDE Plot: Fuel Consumption")
plt.show()

# Barplot 
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Target Type', y='Mission Cost (billion USD)', estimator=np.mean,
            errorbar=None, hue='Target Type', palette='coolwarm', legend=False)
plt.title("Barplot: Avg Mission Cost by Target Type")
plt.xticks(rotation=30)
plt.show()

# === STATISTICAL TESTS ===

# Chi-square test
print("\n=== Chi-Square Test: Launch Vehicle vs Mission Type ===")
chi_data = pd.crosstab(df['Launch Vehicle'], df['Mission Type'])
chi2_result = chi2_contingency(chi_data)
print(f"Chi2 = {chi2_result[0]:.4f}, p-value = {chi2_result[1]:.4f}")

# T-test: Scientific Yield (Research vs Colonization)
print("\n=== T-Test: Scientific Yield (Research vs Colonization) ===")
t_res = ttest_ind(
    df[df['Mission Type'] == 'Research']['Scientific Yield (points)'],
    df[df['Mission Type'] == 'Colonization']['Scientific Yield (points)'],
    equal_var=False
)
print(f"T-statistic = {t_res.statistic:.4f}, p-value = {t_res.pvalue:.4f}")

# Z-test: Mission Cost (Moon vs Exoplanet)
print("\n=== Z-Test: Mission Cost (Moon vs Exoplanet) ===")
z_stat, z_pval = ztest(
    df[df['Target Type'] == 'Moon']['Mission Cost (billion USD)'],
    df[df['Target Type'] == 'Exoplanet']['Mission Cost (billion USD)']
)
print(f"Z-statistic = {z_stat:.4f}, p-value = {z_pval:.4f}")
