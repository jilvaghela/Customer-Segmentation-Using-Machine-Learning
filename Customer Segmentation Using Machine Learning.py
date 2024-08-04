#!/usr/bin/env python
# coding: utf-8

# # customer segmentation

# In[19]:


# Import required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# In[4]:


customersdata = pd.read_csv("Mall_Customers.csv")
customersdata


# K-means model

# In[5]:


kmeans_model = KMeans(init='k-means++',  max_iter=400, random_state=42)


# Train the model

# In[6]:


kmeans_model.fit(customersdata[['CustomerID','Annual Income (k$)',
'Spending Score (1-100)']])


# K means model for different values of K

# In[7]:


def try_different_clusters(K, data):

    cluster_values = list(range(1, K+1))
    inertias=[]

    for c in cluster_values:
        model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)

    return inertias


# output for k values between 1 to 12 

# In[8]:


outputs = try_different_clusters(12, customersdata[['CustomerID','Annual Income (k$)','Spending Score (1-100)']])
distances = pd.DataFrame({"clusters": list(range(1, 13)),"sum of squared distances": outputs})


# In[9]:


figure = go.Figure()
figure.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

figure.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),
                  xaxis_title="Number of clusters",
                  yaxis_title="Sum of squared distances",
                  title_text="Finding optimal number of clusters using elbow method")
figure.show()


# Optimal value of K = 5

# In[10]:


kmeans_model_new = KMeans(n_clusters = 5,init='k-means++',max_iter=400,random_state=42)

kmeans_model_new.fit_predict(customersdata[['CustomerID','Annual Income (k$)','Spending Score (1-100)']])


# # Visualizing customer segments

# In[11]:


cluster_centers = kmeans_model_new.cluster_centers_
data = np.expm1(cluster_centers)
points = np.append(data, cluster_centers, axis=1)
points


# In[12]:


points = np.append(points, [[0], [1], [2], [3], [4]], axis=1)
customersdata["clusters"] = kmeans_model_new.labels_


# Add 'clusters' to customers data

# In[20]:


customersdata.head()


# In[21]:


# visualize clusters
figure = px.scatter_3d(customersdata,
                    color='clusters',
                    x="CustomerID",
                    y="Annual Income (k$)",
                    z="Spending Score (1-100)",
                    category_orders = {"clusters": ["0", "1", "2", "3", "4"]}
                    )
figure.update_layout()
figure.show()


# Distribution of Age

# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
sns.histplot(customersdata['Age'], kde=True, bins=10)
plt.title('Normal Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[23]:


plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', data=customersdata, hue='Gender', palette=['#1f77b4', '#ff7f0e'])
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[17]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

# List of gender values and labels
genders = ['Male', 'Female']
labels = ['Males', 'Females']

for ax, gender, label in zip(axes, genders, labels):
    
    # Plotting the distribution
    sns.histplot(customersdata['Spending Score (1-100)'][customersdata['Gender'] == gender], kde=True, bins=10, ax=ax)
    ax.set_title(f'Distribution of Spending Score for {label}')
    ax.set_xlabel('Spending Score (1-100)')
    ax.set_ylabel('Frequency')

# Adjust layout and show plot
plt.tight_layout()
plt.show()


# In[18]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
genders = ['Male', 'Female']
labels = ['Males', 'Females']
for ax, gender, label in zip(axes, genders, labels):
    sns.histplot(customersdata['Annual Income (k$)'][customersdata['Gender'] == gender], kde=True, bins=10, ax=ax)
    ax.set_title(f'Distribution of Annual Income for {label}')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[ ]:




