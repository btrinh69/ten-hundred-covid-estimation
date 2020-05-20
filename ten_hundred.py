#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv

"""
takes in a string with a path to a CSV file and returns the data 
(without the lat/long columns but retaining all other columns) 
in a single structure
@param filepath: the path to the CSV file
@return a list of dictionary (each dict is the info of the death
num in a country)
"""
def load_data(filepath):
    data = []
    with open(filepath, newline = '') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            del row["Lat"]
            del row["Long"]
            data.append(row)
    return data


# In[2]:


from datetime import datetime
import copy
import math

"""
Take the data of a country in the form of dict, calculate the n/10
and n/100 days
@param time_series: a dict of info about each country
@return a tuple contain n/10 and n/100 days
"""
def calculate_x_y(time_series):
    temp = copy.deepcopy(time_series)
    del temp['Province/State']
    del temp['Country/Region']
    date = list(temp.keys())
    date.sort(key=lambda date: datetime.strptime(date, "%m/%d/%y"), reverse = True)
    if (int(time_series[date[0]])==0):
        return math.nan, math.nan
    x = -1
    y = -1
    for i in range(1, len(date)):
        if int(time_series[date[i]]) <= int(time_series[date[0]])/10:
            x = i
            break
    if x == -1:
        return math.nan, math.nan
    for i in range(1, len(date)):
        if int(time_series[date[i]]) <= int(time_series[date[0]])/100:
            y = i
            break
    if y == -1:
        y = math.nan
    return (x, y-x)


# In[3]:


"""
helper function to calculate the n/10 and n/100 values for the
whole dataset
@return the list of n/10 and n/100 values
"""
def x_y(dataset):
    cor = []
    for i in dataset:
        temp = calculate_x_y(i)
        if not (math.isnan(temp[0]) or math.isnan(temp[1])):
            cor.append(temp)
    return cor


# In[4]:


"""
Helper method to calculate the Euclidean distance
"""
def d(p1, p2):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

"""
Calculate the flattened upper triangle distance matrix
(without the diagonal)
"""
def distance(cor):
    m = []
    for p1 in range(len(cor)):
        for p2 in range(p1, len(cor)):
            if p1==p2:
                continue
            dist = d(cor[p1], cor[p2])
            m.append((dist, p1, p2))
    m.sort()
    return m


# In[5]:


import numpy as np

"""
performs single linkage hierarchical agglomerative clustering on the regions
with the (x,y) feature representation,  and returns a data structure representing 
the clustering.
@param dataset: the loaded dataset
@return An (m-1) by 4 matrix Z. At the i-th iteration, clusters with indices Z[i, 0]
and Z[i, 1] are combined to form cluster m + i. A cluster with an index less than m 
corresponds to one of the m original observations. The distance between clusters 
Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents the 
number of original observations in the newly formed cluster.
"""
def hac(dataset):    
    cor = x_y(dataset)
    dist = distance(cor)
    cluster = []
    while len(dist)>0:
        i = dist.pop(0)
        n1 = 1
        n2 = 1
        if i[1] >= len(cor):
            n1 = cluster[i[1]-len(cor)][3]
        if i[2] >= len(cor):
            n2 = cluster[i[2]-len(cor)][3]
        cluster.append([i[1], i[2], i[0], n1+n2])
        if cluster[-1][3]==len(cor):
            break
                
        cl_idx = len(cor)+len(cluster)-1
        j = 0
        while j < len(dist):
            if dist[j][1]==i[1] or dist[j][1]==i[2]:
                dist[j] = (dist[j][0], cl_idx, dist[j][2])
            if dist[j][2]==i[1] or dist[j][2]==i[2]:
                dist[j]= (dist[j][0], dist[j][1], cl_idx)
            if dist[j][1]==dist[j][2]:
                dist.pop(j)
            else:
                if dist[j][1] > dist[j][2]:
                    dist[j] = (dist[j][0], dist[j][2], dist[j][1])
                j += 1
        dist.sort()
        
    return np.asmatrix(cluster)    

