#IMPORTING NECESSARY LIBRARIES
import imp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
from math import sqrt
import statsmodels.api as sm
import os
import math
from datetime import datetime
from datetime import timedelta

#IMPORTING DATASET
df=pd.read_csv(r'C:\Users\SAKSHAM\Desktop\Coding\Sample DA Practise\Analysis on Supermarket Sales\supermarket_sales - Sheet1.csv')
df.head(5)

#EXPLORING THE DATA

#df.info()
"""With this we can figure out their are no null values present in the dataset
This will remove the duplicates"""

df.duplicated(subset=['Invoice ID', 'Branch', 'City', 'Customer type', 'Gender',
       'Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Total', 'Date',
       'Time', 'Payment', 'cogs', 'gross margin percentage', 'gross income',
       'Rating'])

df['Invoice ID'].nunique()

"""Let's see the best selling Product line for each branch"""
pd.pivot_table(df, index='Product line', columns='Branch', values='gross income',aggfunc='count')

"""We can conclude that
Branch A: Home and Lifestyle
Branch B: Fashion Accessories and Sports & Travel
Branch C: Food and beverages has the highest and Fashion Accessories just comes second"""


"""Let's find out What the best selling branch?"""

best_selling_branch = df.groupby('Branch')[['gross income']].mean().reset_index()
best_selling_branch_plot=sns.barplot(x='Branch',y='gross income',data=best_selling_branch)
#plt.show()
"""Therefore C Branch has the best gross income"""


"""Now let's a go a bit deeper """

"""C branch has the most gross income"""

c_info = df.groupby('Branch')[['Rating']].mean().reset_index()
#print(c_info)

"""Let's analyze more about Branch C """
C = df.groupby('Branch')
branch_c = C.get_group('C')
branch_c = pd.DataFrame(branch_c)

"""Day to Day Analysis"""
df['Date']=pd.to_datetime(df['Date'])
df['weekday']=df['Date'].dt.day_name()
w = df.groupby('weekday')[['gross income']].sum().reset_index()
#sns.barplot(x='weekday',y='gross income',data=w)
#plt.xticks(rotation=45)


"""With above analysis we can conclude that Saturday produces the most revenue"""

df.nunique()

"""Let's find what kind of customers are regular"""
#sns.catplot(x='Customer type',kind='count',data=branch_c)
#sns.catplot(x='Payment',kind='count',data=branch_c)
#sns.catplot(x='Customer type',hue='Payment',kind='count',data=branch_c)
#plt.show()

"""Their are two kind of customers Normal Customers and Customers applied for membership
those who are just regular majority of the time tend to pay via cash 
followed by E-Wa
llet whereas customers with membership pays either through credit card or cash"""

"""Let's find out more about customers"""
sns.catplot(x='Gender',kind='count',data=branch_c)
plt.plot()

"""Mostly customers are Female"""


"""Using Statistics for deeper analysis of Branch C"""

x =  branch_c[['Unit price','Quantity','Total','Rating','cogs','Tax 5%']]
y= branch_c[['gross income']]

x = sm.add_constant(x)
model=sm.OLS(y,x).fit()
#print(model.summary())

"""To find the correlation between them"""
df_num = df[['Unit price','Quantity','Total','cogs','gross income','Rating']]
print(df_num.corr())