#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from scipy import stats

#LOAD DATASET (FIXED)
df = pd.read_csv(
    r"C:\Users\DELL\Downloads\42_Cases_under_crime_against_women.csv",
    encoding='latin1',
    engine='python',
    on_bad_lines='skip'
)

print("\n🔹Columns:")
print(df.columns)


#DATA CLEANING
df = df.dropna()
df = df.drop_duplicates()


#NUMERIC DATA
num_df = df.select_dtypes(include=np.number)


#NUMPY OPERATIONS
print("\nMean:\n", np.mean(num_df))
print("\nMax:\n", np.max(num_df))
print("\nMin:\n", np.min(num_df))


#EDA 
# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Distribution Plot
plt.figure(figsize=(8,5))
sns.histplot(num_df.iloc[:,0], kde=True)
plt.title("Distribution of Crime Data")
plt.show()

#VISUALIZATION
# 1. Year-wise crime trend
year_col = [col for col in df.columns if 'year' in col.lower()][0]

year_data = df.groupby(year_col).sum(numeric_only=True)

plt.figure(figsize=(10,5))
plt.plot(year_data.index, year_data.sum(axis=1), marker='o')
plt.title("Total Crimes Over Years")
plt.xlabel("Year")
plt.ylabel("Total Crimes")
plt.show()

# 2. Top 5 Crime Categories
crime_totals = num_df.sum().sort_values(ascending=False).head(5)

plt.figure(figsize=(8,5))
plt.bar(crime_totals.index, crime_totals.values)
plt.title("Top 5 Crime Categories")
plt.xticks(rotation=45)
plt.ylabel("Total Cases")
plt.show()

# 3. Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(num_df.iloc[:,0], num_df.iloc[:,1])
plt.xlabel(num_df.columns[0])
plt.ylabel(num_df.columns[1])
plt.title("Scatter Plot")
plt.show()

#STATISTICS
# Correlation
corr = num_df.iloc[:,0].corr(num_df.iloc[:,1])
print("\nCorrelation:", corr)

# T-Test
t_stat, p_val = stats.ttest_ind(num_df.iloc[:,0], num_df.iloc[:,1])
print("T-test:", t_stat, p_val)

# Shapiro Test
shapiro = stats.shapiro(num_df.iloc[:,0].sample(100))
print("Shapiro Test:", shapiro)


#MACHINE LEARNING 
# Features & Target
X = num_df.iloc[:, :-1]
y = num_df.iloc[:, -1]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Results:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


#RESULT VISUALIZATION
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
