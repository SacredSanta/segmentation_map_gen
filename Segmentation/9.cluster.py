#%% practice code

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import os
current_path = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv("Mall_Customers.csv")
print(df)
data = df[['Annual Income (k$)', 'Spending Score (1-100)']]

k = 3
model = KMeans(n_clusters = k, init = 'k-means++', random_state = 10)

def elbow(X):
    sse = []
    for i in range(1,11):
        km = KMeans(n_clusters=i,init='k-means++',random_state=0)
        km.fit(X)
        sse.append(km.inertia_)

    plt.plot(range(1,11),sse,marker='o')
    plt.xlabel('# of clusters')
    plt.ylabel('Inertia')
    plt.show()
elbow(data)

df['cluster'] = model.fit_predict(data)

final_centroid = model.cluster_centers_
print(final_centroid)

plt.figure(figsize=(8,8))
for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'Annual Income (k$)'], df.loc[df['cluster'] == i, 'Spending Score (1-100)'], label = 'cluster' + str(i))

plt.scatter(final_centroid[:,0], final_centroid[:,1],s=50,c='violet',marker = 'x', label = 'Centroids')
plt.legend()
plt.title(f'K={k} results',size = 15)
plt.xlabel('Annual Income',size = 12)
plt.ylabel('Spending Score',size = 12)
plt.show()


#%% Init data

import struct 
import pandas as pd 
from joblib import Parallel, delayed, cpu_count

binfile = '240826_202857-1_ba133_11231588_29m39s_thres1000'

num_cores = cpu_count()

i = 1000000
bytes_count = [8,8,4,4,4,4]
bytes_type = ['<2f','<2i','<f','<f','<f','<f']
cols = ['time_stamp', 'event_number', 'ch1', 'ch2', 'ch3', 'ch4']
#df = pd.DataFrame(columns=cols)
pd_idx = 0

def data_stream(nn, filepath):
    with open(f"{filepath}.bin", 'rb') as file:
        file.seek(nn*32)
        new_row = {}
        
        print( )
        for ii in range(6):
            val_hx = file.read(bytes_count[ii])
            val = struct.unpack(bytes_type[ii], val_hx)[0]
            new_row[cols[ii]] = val
        
    return new_row



# Read the entire file
results = Parallel(n_jobs=num_cores, verbose=10)(delayed(data_stream)(num, binfile) for num in range(i))
df = pd.DataFrame(results) 
    
'''
with open(f'{binfile}.bin', 'rb') as file:
    while i > -1:
        if i % 10000 == 0:
            print(i, "data left..")
        i -= 1
        new_row = {}
        for ii in range(6):
            val_hx = file.read(bytes_count[ii])
            val = struct.unpack(bytes_type[ii], val_hx)[0]
            new_row[cols[ii]] = val
        df.loc[pd_idx] = new_row    
        pd_idx += 1
'''      
            
df['x'] = df['ch2']-df['ch1'] / df['ch2']+df['ch1']
df['y'] = df['ch4']-df['ch3'] / df['ch4']+df['ch3']
df = df.dropna()

# %% Cluster
           
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


data = df[['x', 'y']]
data = data.dropna()

k = 38*38

model = KMeans(n_clusters = k, init = 'k-means++', random_state = 10)

def elbow(X):
    sse = []
    
    for i in range(1,11):
        km = KMeans(n_clusters=i,init='k-means++',random_state=0)
        km.fit(X)
        sse.append(km.inertia_)

    plt.plot(range(1,11),sse,marker='o')
    plt.xlabel('# of clusters')
    plt.ylabel('Inertia')
    plt.show()
    
elbow(data)

df['cluster'] = model.fit_predict(data)

final_centroid = model.cluster_centers_
print(final_centroid)

plt.figure(figsize=(8,8))
for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'x'], df.loc[df['cluster'] == i, 'y'], label = 'cluster' + str(i))

plt.scatter(final_centroid[:,0], final_centroid[:,1],s=50,c='violet',marker = 'x', label = 'Centroids')
plt.legend()
plt.title(f'K={k} results',size = 15)
plt.xlabel('x',size = 12)
plt.ylabel('y',size = 12)
plt.show()

#%%
plt.scatter()

