# EDA-task-5
1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# For visual clarity
# %matplotlib inline
sns.set(style="whitegrid")

# 2. Load Dataset
df = pd.read_csv('/content/train_and_test2.csv')
df.head()

# Structure of the dataset
df.info()

# Summary statistics
df.describe()

# Null values
df.isnull().sum()

# Unique values
df.nunique()

# Survival count
sns.countplot(x='2urvived', data=df)
plt.title("Survival Count")
plt.xlabel("Survived (1 = Yes, 0 = No)")
plt.ylabel("Number of Passengers")
plt.show()

# Class distribution
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

# Gender distribution
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

# Age distribution
df['Age'].hist(bins=30, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
# Fare distribution
df['Fare'].hist(bins=30, edgecolor='black')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# Survival by Sex
sns.countplot(x='Sex', hue='2urvived', data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by Pclass
sns.countplot(x='Pclass', hue='2urvived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Boxplot: Age vs Survived
sns.boxplot(x='2urvived', y='Age', data=df)
plt.title("Age Distribution by Survival")
plt.show()

# Boxplot: Fare vs Survived
sns.boxplot(x='2urvived', y='Fare', data=df)
plt.title("Fare Distribution by Survival")
plt.show()

# Age vs Fare Scatterplot
sns.scatterplot(x='Age', y='Fare', hue='2urvived', data=df)
plt.title("Age vs Fare (Colored by Survival)")
plt.show()

# Only numeric columns
corr = df.corr()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df[['2urvived', 'Age', 'Fare', 'Pclass']], hue='2urvived')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Display as plain text
summary = """
KEY INSIGHTS:

1. Female passengers had a much higher survival rate than males.
2. Passengers in 1st class had better chances of survival.
3. Younger passengers (children) also had higher survival rates.
4. Fare shows a positive correlation with survival â€“ higher fare passengers tended to survive.
5. Age and Fare are slightly correlated with each other.
6. 'Cabin' and 'Ticket' might need additional feature engineering due to high cardinality or missing values.
"""
print(summary)


📌 SUMMARY: How to Create a PDF Report of Findings (EDA)

✅ 1. Using Jupyter Notebook

Perform EDA using Pandas, Seaborn, Matplotlib.

Add markdown cells for observations and explanations.

Go to File > Export Notebook As > PDF to generate the report.

Best for fast reporting with all code, plots, and comments included.


✅ 2. Using Python Script + fpdf

Use fpdf to programmatically create a custom PDF.

Save plots using plt.savefig().

Add text insights and images to the PDF.


Example:

from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, "EDA Report", ln=True, align="C")
pdf.image("plot.png", x=10, y=30, w=180)
pdf.output("report.pdf")

✅ 3. Manual Method

Save plots as images.

Write observations in Word / Google Docs.

Insert images and export as PDF.

