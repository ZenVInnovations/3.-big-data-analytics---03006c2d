# -*- coding: utf-8 -*-

from google.colab import files
uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load uploaded CSV
df = pd.read_csv("road_accidents.csv")

# Show basic info
print("Data Head:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# Drop rows with missing values in important columns
df.dropna(subset=['Accident_Severity', 'Number_of_Vehicles'], inplace=True)

# Convert Date to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract Month
df['Month'] = df['Date'].dt.month

# 1. Monthly accident count
monthly_accidents = df['Month'].value_counts().sort_index()
sns.barplot(x=monthly_accidents.index, y=monthly_accidents.values, palette='viridis')
plt.title("Monthly Road Accidents")
plt.xlabel("Month")
plt.ylabel("Accidents")
plt.show()

# 2. Severity distribution
severity_counts = df['Accident_Severity'].value_counts()
sns.barplot(x=severity_counts.index, y=severity_counts.values, palette='rocket')
plt.title("Accident Severity")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.show()

# 3. Number of vehicles involved
sns.histplot(df['Number_of_Vehicles'], bins=10, kde=True)
plt.title("Vehicles Involved in Accidents")
plt.xlabel("Number of Vehicles")
plt.ylabel("Frequency")
plt.show()

# 4. Accidents by weather
weather_accidents = df['Weather_Conditions'].value_counts()
sns.barplot(x=weather_accidents.index, y=weather_accidents.values, palette='coolwarm')
plt.title("Accidents by Weather")
plt.xlabel("Weather")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# 5. Correlation heatmap
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='magma')
plt.title("Correlation Matrix")
plt.show()