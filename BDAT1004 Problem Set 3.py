#!/usr/bin/env python
# coding: utf-8

# # Question 1 
# 
# Introduction:
# Special thanks to: https://github.com/justmarkham for sharing the dataset and
# materials.
# Occupations
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address.
# Step 3. Assign it to a variable called users
# Step 4. Discover what is the mean age per occupation
# Step 5. Discover the Male ratio per occupation and sort it from the most to the least
# Step 6. For each occupation, calculate the minimum and maximum ages
# Step 7. For each combination of occupation and sex, calculate the mean age
# Step 8. For each occupation present the percentage of women and men

# Step 1 : Import the necessary libraries 

# In[8]:


import pandas as pd
import urllib


# Step 2. Import the dataset from this address.

# In[9]:


url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user' 


# Step 3. Assign it to a variable called users

# In[10]:


users = pd.read_csv(url)
arr = users['user_id|age|gender|occupation|zip_code'].str.split('|',expand = True)
users['user_id'] = arr[0]
users['age']=arr[1]
users['gender']=arr[2]
users['occupation']=arr[3]
users['zip_code']=arr[4]
users.drop(columns = ['user_id|age|gender|occupation|zip_code'], inplace = True)
users['age']=users['age'].astype(int)
users.head()


# Step 4. Discover what is the mean age per occupation

# In[12]:


users.groupby('occupation').mean()


# Step 5. Discover the Male ratio per occupation and sort it from the most to the least

# In[13]:


maleRatio = users.pivot_table(index='occupation', columns='gender', aggfunc='size', fill_value=0)

total = maleRatio[['F', 'M']].sum(axis=1)
maleRatio['MaleRatio'] = 100 * maleRatio['M'] / total

maleRatio = maleRatio['MaleRatio'].sort_values(ascending=False)
# result
maleRatio


# Step 6. For each occupation, calculate the minimum and maximum ages

# In[15]:


users.groupby('occupation')['age'].agg(['min','max'])


# Step 7. For each combination of occupation and sex, calculate the mean age

# In[16]:


users.groupby(['occupation', 'gender'])['age'].mean()


# Step 8. For each occupation present the percentage of women and men

# In[18]:


GenderOccupation = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})
occupationCount = users.groupby(['occupation']).agg('count')
occupationCount
GenderPercentage = GenderOccupation.div(occupationCount, level = "occupation") * 100
GenderPercentage.loc[:,'gender']


# # Question 2:
# 
# Euro Teams
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address
# Step 3. Assign it to a variable called euro12
# Step 4. Select only the Goal column
# Step 5. How many team participated in the Euro2012?
# Step 6. What is the number of columns in the dataset?
# Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them
# to a dataframe called discipline
# Step 8. Sort the teams by Red Cards, then to Yellow Cards
# Step 9. Calculate the mean Yellow Cards given per Team
# Step 10. Filter teams that scored more than 6 goalsStep 11. Select the teams that start
# with G
# Step 12. Select the first 7 columns
# Step 13. Select all columns except the last 3
# Step 14. Present only the Shooting Accuracy from England, Italy and Russia

# Step 1. Import the necessary libraries

# In[20]:


import pandas as pd


# Step 2. Import the dataset from this address.

# In[23]:


url1 = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv'


# Step 3. Assign it to a variable called euro12

# In[24]:


euro12 = pd.read_csv(url1)
euro12.head()


# Step 4. Select only the Goal column

# In[25]:


euro12["Goals"]


# Step 5. How many team participated in the Euro2012?

# In[26]:


teamsCount = len(euro12)
teamsCount


# Step 6. What is the number of columns in the dataset?

# In[27]:


euro12.shape[1]


# Step 7. View only the columns Team, Yellow Cards and Red Cards and assign them to a dataframe called discipline

# In[28]:


discipline=pd.DataFrame({'Team':euro12['Team'], 'Yellow Cards': euro12['Yellow Cards'],'Red Cards': euro12['Red Cards']})
discipline


# Step 8. Sort the teams by Red Cards, then to Yellow Cards

# In[29]:


discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending = True)


# Step 9. Calculate the mean Yellow Cards given per Team

# In[30]:


perteam= pd.DataFrame(euro12.groupby(['Team'])['Yellow Cards'].mean())
perteam


# Step 10. Filter teams that scored more than 6 goalsStep

# In[31]:


euro12[euro12['Goals'] > 6]


# Step 11. Select the teams that start with G

# In[32]:


gteams=euro12[euro12['Team'].str.startswith('G')]
gteams


# Step 12. Select the first 7 columns

# In[33]:


euro12[euro12.columns[0:7]]


# Step 13. Select all columns except the last 3

# In[34]:


euro12.iloc[:,0:-3]


# Step 14. Present only the Shooting Accuracy from England, Italy and Russia

# In[35]:


Countries = euro12[euro12.Team.isin(['England','Italy','Russia'])]
Countries[['Team', 'Shooting Accuracy']]


# # Question 3:
# 
# Housing
# Step 1. Import the necessary libraries
# Step 2. Create 3 differents Series, each of length 100, as follows:
# • The first a random number from 1 to 4
# • The second a random number from 1 to 3
# • The third a random number from 10,000 to 30,000
# Step 3. Create a DataFrame by joinning the Series by column
# Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter
# Step 5. Create a one column DataFrame with the values of the 3 Series and assign it
# to 'bigcolumn'
# Step 6. Ops it seems it is going only until index 99. Is it true?
# Step 7. Reindex the DataFrame so it goes from 0 to 299

# In[37]:


#Step 1. Import the necessary libraries
import pandas as pd
import numpy as np


# In[51]:


#Step 2. Create 3 differents Series, each of length 100, as follows:
#• The first a random number from 1 to 4
#• The second a random number from 1 to 3
#• The third a random number from 10,000 to 30,000

series1 = pd.Series(np.random.randint(1, high=5, size=100, dtype='l'))     #defining series 1,2, and 3
series2 = pd.Series(np.random.randint(1, high=4, size=100, dtype='l'))
series3 = pd.Series(np.random.randint(10000, high=30001, size=100, dtype='l'))

print ("series1 is =", series1)    #printing series 1,2, and 3
print ("series 2 is =", series2)
print("series 3 ia =", series3)


# In[52]:


#Step 3. Create a DataFrame by joinning the Series by column

data_frame = pd.concat([series1, series2, series3], axis=1)   #joining 1,2, and 3 series
print(data_frame)


# In[53]:


#Step 4. Change the name of the columns to bedrs, bathrs, price_sqr_meter
data_frame.columns = ['bedrs','bathrs','price_sqr_meter']
data_frame.head()


# In[54]:


#Step 5. Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'

# join concat the values
column = pd.concat([series1, series2, series3], axis=0)

# it is still a Series, so we need to transform it to a DataFrame
column = column.to_frame()
print (column)
 
column


# In[55]:


#Step 6. Ops it seems it is going only until index 99. Is it true?

if (max(column.index)==99):
    print('True')
else:
    print('False')


# In[56]:


#Step 7. Reindex the DataFrame so it goes from 0 to 299

new_index=[x for x in range(0,300)]
column = pd.concat([series1, series2, series3], axis=0,ignore_index=True)
column = column.to_frame()
column=column.reindex(new_index)
print("column is") 
column


# # Question 4 :
# Wind Statistics
# The data have been modified to contain some missing values, identified by NaN.
# Using pandas should make this exercise easier, in particular for the bonus question.
# You should be able to perform all of these operations without using a for loop or
# other looping construct.
# The data in 'wind.data' has the following format:
# Yr Mo Dy RPT VAL ROS KIL SHA BIR DUB CLA MUL CLO BEL
# MAL
# 61 1 1 15.04 14.96 13.17 9.29 NaN 9.87 13.67 10.25 10.83 12.58 18.50 15.04
# 61 1 2 14.71 NaN 10.83 6.50 12.62 7.67 11.50 10.04 9.79 9.67 17.54 13.83
# 61 1 3 18.50 16.88 12.33 10.13 11.17 6.17 11.25 NaN 8.50 7.67 12.75 12.71
# 
# The first three columns are year, month, and day. The remaining 12 columns are
# average windspeeds in knots at 12 locations in Ireland on that day.
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from the attached file wind.txt
# Step 3. Assign it to a variable called data and replace the first 3 columns by a proper
# datetime index.
# Step 4. Year 2061? Do we really have data from this year? Create a function to fix it
# and apply it.
# Step 5. Set the right dates as the index. Pay attention at the data type, it should be
# datetime64[ns].
# Step 6. Compute how many values are missing for each location over the entire
# record.They should be ignored in all calculations below.
# Step 7. Compute how many non-missing values there are in total.
# Step 8. Calculate the mean windspeeds of the windspeeds over all the locations and
# all the times.
# A single number for the entire dataset.
# Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean
# windspeeds and standard deviations of the windspeeds at each location over all the
# days
# A different set of numbers for each location.
# Step 10. Create a DataFrame called day_stats and calculate the min, max and mean
# windspeed and standard deviations of the windspeeds across all the locations at each
# day.
# A different set of numbers for each day.
# Step 11. Find the average windspeed in January for each location.
# Treat January 1961 and January 1962 both as January.
# Step 12. Downsample the record to a yearly frequency for each location.
# Step 13. Downsample the record to a monthly frequency for each location.
# Step 14. Downsample the record to a weekly frequency for each location.
# Step 15. Calculate the min, max and mean windspeeds and standard deviations of the
# windspeeds across all locations for each week (assume that the first week starts on B
# January 2 1961) for the first 52 weeks

# In[57]:


#Step 1. Import the necessary libraries
import pandas as pd
import numpy as np
import datetime


# In[58]:


#Step 2. Import the dataset from this address
read_data = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/Wind_Stats/wind.data", delim_whitespace=True)
read_data.head(10)


# In[59]:


# step 3
# Assign it to a variable called data and replace the first 3 columns by a proper datetime index

data = read_data['Yr'].astype(str) + '-' + read_data['Mo'].astype(str) + '-' + read_data['Dy'].astype(str)
read_data.insert(0, 'Date1', data)
data['Date1'] =  pd.to_datetime(read_data['Date1'])
data = read_data.drop(columns = ['Yr', 'Mo', 'Dy'])

data


# In[60]:


data = read_data.drop(columns = ['Date1'])
data


# In[61]:


#step 4
# Year 2061? Do we really have data from this year? Create a function to fix it and apply it.

#creating a function
def add_2061(x):
    return '2061'+ str(x)

data['Yr'] = data['Yr'].apply(add_2061)
data


# In[62]:


#step 6. Compute how many values are missing for each location over the entire record.They should be ignored in all calculations below.

data.isnull().sum()


# In[63]:


#step 7 Compute how many non-missing values there are in total
data.notnull().sum()


# In[64]:


#step 8  Calculate the mean windspeeds of the windspeeds over all the locations and all the times.
data.mean(axis=0)


# In[65]:


#Step 9. Create a DataFrame called loc_stats and calculate the min, max and mean windspeeds and standard deviations of the windspeeds at each location over all the days A different set of numbers for each location.

loc_stats=data.describe(percentiles=[])
loc_stats.head(10)


# In[66]:


# Step 10. Create a DataFrame called day_stats and calculate the min, max and mean windspeed and standard deviations of the windspeeds across all the locations at each day.
#A different set of numbers for each day.

day_stats = pd.DataFrame()
day_stats['mean'] = read_data.mean(axis = 1)
day_stats['min'] = read_data.min(axis = 1)
day_stats['max'] = read_data.max(axis = 1)
day_stats['std'] = read_data.std(axis = 1)
round(day_stats, 1)


# # Question 5 :
#     
# Step 1. Import the necessary libraries
# Step 2. Import the dataset from this address.
# Step 3. Assign it to a variable called chipo.
# Step 4. See the first 10 entries
# Step 5. What is the number of observations in the dataset?
# Step 6. What is the number of columns in the dataset?
# Step 7. Print the name of all the columns.
# Step 8. How is the dataset indexed?
# Step 9. Which was the most-ordered item?
# Step 10. For the most-ordered item, how many items were ordered?
# Step 11. What was the most ordered item in the choice_description column?
# Step 12. How many items were orderd in total?
# Step 13.
# • Turn the item price into a float
# • Check the item price type
# • Create a lambda function and change the type of item price
# • Check the item price type
# Step 14. How much was the revenue for the period in the dataset?
# Step 15. How many orders were made in the period?
# Step 16. What is the average revenue amount per order?
# Step 17. How many different items are sold?

# In[ ]:


#Step 1. Import the necessary libraries
import pandas as pd


# In[68]:


#Step 2. Import the dataset from this address.
link = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"


# In[69]:


#Step 3. Assign it to a variable called chipo.

chipo = pd.read_csv(link, sep='\t')


# In[70]:


#Step 4. See the first 10 entries

chipo.head(10)


# In[71]:


#Step 5. What is the number of observations in the dataset?


chipo.shape[0]


# In[72]:


#Step 6. What is the number of columns in the dataset?
len(chipo.columns)


# In[73]:


# Step 7. Print the name of all the columns.

chipo.columns


# In[74]:


#Step 8. How is the dataset indexed?

chipo.index


# In[75]:


#Step 9. Which was the most-ordered item?

ordered_item = chipo.item_name.value_counts()
ordered_item


# In[76]:


#Step 10. For the most-ordered item, how many items were ordered?

ordered_item[:1]


# In[77]:


#Step 11. What was the most ordered item in the choice_description column?

ch_description = chipo.choice_description.value_counts()
ch_description[:1]


# In[78]:


# Step 12. How many items were orderd in total?

total = chipo.quantity.sum()
total


# In[79]:


#Step 13.
#• Turn the item price into a float
chipo.item_price.str.slice(1).astype(float).head()
chipo.dtypes['item_price']
chipo['item_price'] = chipo.apply(lambda x: float(x['item_price'].replace('$', '')),axis=1)
chipo.dtypes['item_price']


# In[80]:


#Step 14. How much was the revenue for the period in the dataset?

revenue = chipo['item_price'] * chipo['quantity']
revenue.sum()    


# In[81]:


#Step 15. How many orders were made in the period?

orders = chipo['order_id'].nunique()
orders


# In[82]:


#Step 16. What is the average revenue amount per order?

round(revenue.sum()/orders, 3)


# In[83]:


# Step 17. How many different items are sold?
chipo['item_name'].nunique()


# # Question 6 :
# 
# Create a line plot showing the number of marriages and divorces per capita in the
# U.S. between 1867 and 2014. Label both lines and show the legend.
# Don't forget to label your axes!

# In[131]:


#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[132]:


read_data = pd.read_csv(r"C:\Users\hitik\OneDrive\Desktop\us-marriages-divorces-1867-2014.csv")  #reading data from local disk

plt.plot(read_data.Year, read_data['Marriages'], label='Marriages')    #ploting and reading Marriages data
plt.plot(read_data.Year, read_data['Divorces'], label='Divorces')      #ploting and reading Divorces data

plt.title("Divorces and Marriages per capita in U.S. between 1867-2014")    #assigning title, y and x axis label to chart
plt.ylabel(" Number of Marriages/Divorces");
plt.xlabel("Year")

plt.legend();


# # Question 7 :
# 
# Create a vertical bar chart comparing the number of marriages and divorces per
# capita in the U.S. between 1900, 1950, and 2000.
# Don't forget to label your axes!
# 

# In[129]:


colours = ['#C4E57D','#A4CCD9']
data1['Marriages'].value_counts().sort_index().plot.bar(color=colours)

data1['Divorces'].value_counts().sort_index().plot.bar(color=colours)

data1.plot.bar(title="Marriages and Divorces per Capita")


# # Question 8 :
# Create a horizontal bar chart that compares the deadliest actors in Hollywood. Sort
# the actors by their kill count and label each bar with the corresponding actor's name.
# Don't forget to label your axes!

# In[ ]:


#importing packages
import pandas as pd
import matplotlib.pyplot as plt


# In[133]:


kill_data = pd.read_csv(r"C:\Users\hitik\Downloads\actor_kill_counts.csv")   #reading actor kill counts file from local disk
kill_data = kill_data.sort_values('Count')

plt.barh(kill_data["Actor"], kill_data.Count)

#printing title
plt.title('Kill count of deadliest actors in Hollywood')

#labelling x and y axis
plt.xlabel('Kills Count')
plt.ylabel('Actor')


plt.show()


# # Question 9 :
# 
# Create a pie chart showing the fraction of all Roman Emperors that were
# assassinated.
# Make sure that the pie chart is an even circle, labels the categories, and shows the
# percentage breakdown of the categories.

# In[134]:


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[135]:


#reading data from local disk
roman_data = pd.read_csv(r"C:\Users\hitik\Downloads\roman-emperor-reigns.csv")
roman= pd.DataFrame({'count' : roman_data.groupby("Cause_of_Death" ).size()}).reset_index() 
labels = roman['Cause_of_Death'].values
sizes = roman['count'].values
x=0.0
y=0.1
explode=[y if item=='Assassinated' else x for item in labels ]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=0, textprops={'fontsize':12})
ax1.axis('equal')  
plt.show()


# # Question 10 :
# 
# Create a scatter plot showing the relationship between the total revenue earned by
# arcades and the number of Computer Science PhDs awarded in the U.S. between
# 2000 and 2009.
# Don't forget to label your axes!
# Color each dot according to its year

# In[142]:


arcade = pd.read_csv(r"C:\Users\hitik\Downloads\arcade-revenue-vs-cs-doctorates.csv")
arcade.head()


# In[143]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


fig, ax = plt.subplots()
ax.scatter(arcade['Total Arcade Revenue (billions)'], arcade['Computer Science Doctorates Awarded (US)'])

ax.set_title('Q10')
ax.set_xlabel('Arcade Rev')
ax.set_ylabel('Doctorates Awarded')

#unable to visualize data with correct color scheme, tried to utilize week 11 instructions unclear


# In[ ]:




