#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on HR Employee

# ## Importing necessary libraries

# In[86]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df = pd.read_csv("D:\pandu\CodeTech\EDA\HR-Employee-Attrition.csv")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[122]:


df.columns


# In[6]:


df.shape


# In[8]:


df.info()


# In[7]:


df.isnull().sum()


# In[123]:


df.head()


# #### No missing values

# In[9]:


numeric_features = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 
                    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 
                    'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
                    'YearsSinceLastPromotion', 'YearsWithCurrManager']


# In[11]:


plt.figure(figsize=(20, 20))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(5, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# In[12]:


categorical_features = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'EnvironmentSatisfaction', 
                        'Gender', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'OverTime', 'PerformanceRating', 
                        'StandardHours', 'WorkLifeBalance']


# In[121]:


# Plotting categorical features
for feature in categorical_features:
    plt.figure(figsize=(2,1))
    sns.countplot(x=df[feature])
    plt.title(f'Count of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# In[25]:


correlation_matrix = df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[93]:


plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Attrition', y='Age')
plt.title('Bar Plot of Age by Attrition')
plt.xlabel('Attrition')
plt.ylabel('Age')
plt.show()


# In[70]:


# Pie chart for 'Department'
plt.figure(figsize=(5,5))
df['Department'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
plt.title('Distribution of Employees Across Departments')
plt.ylabel('')
plt.show()


# In[109]:


plt.figure(figsize=(5,5))
df['WorkLifeBalance'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
plt.title('Distribution of WorkLife Blance of Employees')
plt.show()


# In[108]:


# Pie chart for ''
plt.figure(figsize=(5,5))
df['BusinessTravel'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
plt.title('Distribution of Employees for Business Travel')
plt.ylabel('')
plt.show()


# In[34]:


# Scatter plot for 'Age' vs 'MonthlyIncome'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='MonthlyIncome', data=df,hue='Age')
plt.title('Age vs Monthly Income')
plt.xlabel('Age')
plt.ylabel('Monthly Income')
plt.show()


# In[40]:


# Line plot for 'YearsAtCompany' vs 'YearsSinceLastPromotion'
plt.figure(figsize=(10, 6))
sns.lineplot(x='YearsAtCompany', y='YearsSinceLastPromotion', data=df)
plt.title('Years at Company vs Years Since Last Promotion')
plt.xlabel('Years at Company')
plt.ylabel('Years Since Last Promotion')
plt.show()


# In[94]:


# Line plot for 'YearsAtCompany' vs 'YearsSinceLastPromotion'
plt.figure(figsize=(20, 8))
sns.barplot(x='JobRole', y='YearsInCurrentRole', data=df)
plt.title('Years at Company vs Years In Current Role')
plt.xlabel('Job Role')
plt.ylabel('Years In Current Role')
plt.show()


# In[60]:


# Histogram for 'YearsInCurrentRole'
plt.figure(figsize=(10, 6))
sns.histplot(df['YearsInCurrentRole'], kde=True)
plt.title('Distribution of Years in Current Role')
plt.xlabel('Years in Current Role')
plt.ylabel('Frequency')
plt.show()


# In[56]:


# Box plot for 'MonthlyIncome'
plt.figure(figsize=(20, 11))
sns.boxplot(x='JobRole',y='MonthlyIncome', data=df)
plt.title('Distribution of Monthly Income')
plt.ylabel('Monthly Income')
plt.show()


# In[62]:


plt.figure(figsize=(10, 6))
sns.swarmplot(data=df, x='MaritalStatus', y='Age',hue='MaritalStatus')
plt.title('Age Distribution by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Age')
plt.show()


# In[66]:


# KDE plot for 'HourlyRate' with 'Gender' differentiation
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x="HourlyRate", hue="Gender", multiple="stack")
plt.title('Kernel Density Estimation of Hourly Rate by Gender')
plt.xlabel('Hourly Rate')
plt.ylabel('Density')
plt.show()


# In[85]:


# Point plot for 'Department' vs 'MonthlyIncome'
plt.figure(figsize=(10, 6))
sns.pointplot(x='Department', y='MonthlyIncome', data=df)
plt.title('Point Plot: Monthly Income by Department')
plt.xlabel('Department')
plt.ylabel('Monthly Income')
plt.show()


# In[84]:


plt.figure(figsize=(10,6))
sns.barplot(data=df, x='MonthlyRate', y="EducationField", hue="Gender")


# In[117]:


# Scatter Plot: Relationship between YearsAtCompany and YearsSinceLastPromotion
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='YearsAtCompany', y='YearsSinceLastPromotion')
plt.title('Relationship between Years at Company and Years Since Last Promotion')
plt.xlabel('Years at Company')
plt.ylabel('Years Since Last Promotion')
plt.show()

