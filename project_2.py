#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# In[4]:


get_ipython().system("wget -q -O 'newyork_data.json' https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")
print('Data downloaded!')


# In[6]:


with open('newyork_data.json') as json_data:
    newyork_data = json.load(json_data)


# In[7]:


import numpy as np
import pandas as pd

from bs4 import BeautifulSoup #for parsing HTML and XML documents (parse = analizar gramaticalmente)
import requests


# In[8]:


# Getting the HTML of the page
url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
results = requests.get(url)

# Create a BeautifulSoup objetc (para analizar la data)
soup = BeautifulSoup(results.content, "lxml")


# In[9]:


# Getting the table from URL
table = soup.find_all("table")
    # You can use the find_all() method of soup to extract useful html tags within a webpage.
    # Examples of useful tags: < a > for hyperlinks, < table > for tables, < tr > for table rows, < th > for table headers, and < td > for table cells.

# Transform to DataFrame
df = pd.read_html(str(table))
neighborhood_c = pd.DataFrame(df[0])
neighborhood_c.head(5)


# In[10]:


# first shape code (raw data). We have 180 rows and 3 columns.
neighborhood_c.shape


# In[11]:


# Cleaning the dataset: Ignore cells with a borough that is "Not assigned"
neighborhood_c2 = neighborhood_c.copy()
index_names = neighborhood_c2[neighborhood_c2.Borough == "Not assigned"].index
neighborhood_c2.drop(index_names, inplace = True)
neighborhood_c2 = neighborhood_c2.reset_index(drop=True)
neighborhood_c2.head(10)


# In[ ]:


# Split dataset to have postal code of each neighborhood along with the borough name and neighborhood name, in order to utilize the Foursquare location data
#neighborhood_c3 = neighborhood_c2.copy()
#neighborhood_c3["Neighbourhood"]=neighborhood_c3["Neighbourhood"].str.split(",")
#neighborhood_c3 = neighborhood_c3.apply(pd.Series.explode)
#neighborhood_c3 = neighborhood_c3.reset_index(drop=True)
#neighborhood_c3.head(5)


# In[12]:


# Ignore cells with same value in Borough and Neighbourhood
neighborhood_c3 = neighborhood_c2.copy()
index_names_2 = neighborhood_c3[neighborhood_c3.Borough == neighborhood_c3.Neighbourhood].index
neighborhood_c3.drop(index_names_2, inplace = True)
neighborhood_c3 = neighborhood_c3.reset_index(drop=True)
neighborhood_c3.head(10)


# In[13]:


# Final shape
neighborhood_c3.shape


# ## part 2 getting longitude and latitude

# In[14]:


get_ipython().system('pip install geocoder')
import geocoder


# In[ ]:


# @hidden_cell
# this part was not used. I worked with CSV
#latitude = []
#longitude = []

#for code in neighborhood_c4["Postal Code"]:
   # g = geocoder.arcgis("{}, Toronto, Ontario".format(code))
    #print(code, g.latlng)
   # while (g.latlng is None):
       # g = geocoder.arcgis("{}, Toronto, Ontario".format(code))
      #  print (code, g.latlng)
  #  latlng = g.latlng
   # latitude.append(latlng[0])
   # longitude.append(latlng[1])


# In[15]:


# Getting latitudes and longitudes data from CSV file
latlong_df = r"http://cocl.us/Geospatial_data"
coordinates = pd.read_csv(latlong_df)
coordinates.head(5)


# In[16]:


# Merging neighborhood_c3 and coordinates
df_Toronto = pd.merge(neighborhood_c3, coordinates, how="inner")
df_Toronto = df_Toronto.reset_index(drop=True)
df_Toronto.rename(columns={"Neighbourhood":"Neighborhood"}, inplace=True)
df_Toronto.head(5)


# In[17]:


df_Toronto.shape


# ## part 3 explore cluster in neighbour

# In[47]:


#!conda install -c conda-forge geopy --yes
from geopy.geocoders import Nominatim 

#!pip install --upgrade matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import folium


# In[19]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(df_Toronto['Borough'].unique()),
        df_Toronto.shape[0]))


# In[20]:


#Filtering the original dataframe to work only with borough which contain the word "Toronto".\
df_Toronto_2 = df_Toronto.copy()
df_Toronto_2 = df_Toronto_2[df_Toronto_2['Borough'].str.contains("Toronto")]
df_Toronto_2.head(5)  


# In[21]:


# Counting the number of boroughs retrieved
print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(df_Toronto_2['Borough'].unique()),
        df_Toronto_2.shape[0]))


# In[22]:


# @hidden_cell
CLIENT_ID = 'ZJRZWBIOWJRJWZW04XW2XOOPRYKGAIVRPPEHGZMGVULBCMJR' 
CLIENT_SECRET = 'HT4ZHLOXYEQYGMFHTE35J4SLRHKM5JOFYDZLSXTXI1W3O0ZH'
VERSION = '20180605'
LIMIT = 100


# In[24]:


# Exploring Toronto: Building a function to save venues from Toronto in a radius of 300
def getNearbyVenues(names, latitudes, longitudes, radius=300):
    
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
# Getting 


# In[25]:


# Getting the venues with the function created
Toronto_venues = getNearbyVenues (names =df_Toronto_2["Neighborhood"], latitudes =df_Toronto_2["Latitude"], longitudes = df_Toronto_2["Longitude"])


# In[26]:


Toronto_venues.head()


# In[27]:


# Cleaning dataset: Running the codes below I had some problem adding the column "Neighborhood" because it already existed,so I identified the rows where this value was stored in Venue Category's column and dropped it (rows) before one-hot encode it.
print(Toronto_venues[Toronto_venues["Venue Category"]=="Neighborhood"].index.values)


# In[28]:


# Drop rows with "Neighborhood values" into Venue Category because it is not clear information about categories we need.
index_names2 = Toronto_venues[Toronto_venues["Venue Category"] == "Neighborhood"].index
Toronto_venues.drop(index_names2, inplace = True)
Toronto_venues = Toronto_venues.reset_index(drop=True)
Toronto_venues.head(10)


# In[29]:


Toronto_venues.shape


# In[30]:


# Number of venues in the Dataframe (uniques)
len(Toronto_venues['Venue Category'].unique())


# In[31]:


# Checking number of venues for each neighborhood 
Toronto_venues.groupby('Neighborhood').count()


# In[32]:


# Analizing each Neighborhood with one hot encoding
toronto_onehot = pd.get_dummies(Toronto_venues[['Venue Category']], prefix="", prefix_sep="")
toronto_onehot.head()


# In[33]:


# Adding column Neighborhood to the one-code df.
  # 1st: create a variable with Neighborhood values named "data".
  # 2d: insert the new column with its respective values in variable "data", placed in the 1st column (0).
  # 3rd: reset index.
data = Toronto_venues["Neighborhood"]
toronto_onehot.insert(0, "Neighborhood", data)
toronto_onehot = toronto_onehot.reset_index(drop=True)
toronto_onehot.head()


# In[34]:


# Taking a look to the mean of the frequency of ocurrence of each Neighborhood (this will be used to calculate the distance to the centroids)
toronto_grouped = toronto_onehot.groupby("Neighborhood").mean().reset_index()
toronto_grouped.head(5)


# In[35]:


# Confirming the new size
toronto_grouped.shape


# In[36]:


# Printing each Neighborhood with the top 5 most common venues
num_top_venues = 5

for hood in toronto_grouped["Neighborhood"]:
    print("----"+hood+"----")
    temp = toronto_grouped[toronto_grouped["Neighborhood"] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[37]:


# Transforming top venues into a dataframe: Writting a function to sort the venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[38]:


# Transforming top venues into a dataframe: Running the function with top 5
num_top_venues = 5

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[39]:


#  Cluster Neighborhoods: labels for the model
toronto_grouped_clustering = toronto_grouped.drop("Neighborhood", 1)
toronto_grouped_clustering


# In[42]:


#  Cluster Neighborhoods: Best k - Elbow model (Running K-means with a range of k and plotting - K-means only works with continuos data).
    # source: https://www.aprendemachinelearning.com/k-means-en-python-paso-a-paso/
    # source2: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    # source3: https://predictivehacks.com/k-means-elbow-method-code-for-python/
from scipy.spatial.distance import cdist
distortions = []
mapping = {}
K = range(1,15)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(toronto_grouped_clustering)
    distortions.append(kmeanModel.inertia_)
    mapping[k] = sum(np.min(cdist(toronto_grouped_clustering, kmeanModel.cluster_centers_, 'euclidean'),axis=1)) / toronto_grouped_clustering.shape[0] 


# In[43]:


for key,val in mapping.items(): 
    print(str(key)+' : '+str(val)) 


# In[48]:


plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[49]:


# Cluster Neighborhoods:: K-means to cluster
kclusters = 8
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)
kmeans.labels_.astype(int)
kmeans.labels_[0:10]


# In[50]:


# Cluster Neighborhoods:: Add clusters labels to Dataframe
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_merged = df_Toronto_2 # dataframe with borough that only contain the word "Toronto"

# merge toronto_merged with df_Toronto_2 to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.merge(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_merged.head()


# In[51]:


toronto_merged["Cluster Labels"].unique()


# In[52]:


toronto_merged.shape


# In[53]:


# Getting Toronto, Canada coordinates
address = 'Toronto'

geolocator = Nominatim(user_agent="ca_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto city are {}, {}.'.format(latitude, longitude))


# In[54]:


# Create the map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## examine cluster

# In[55]:


# Cluster 1 (0) - Venues to eat around.
toronto_merged.loc[toronto_merged["Cluster Labels"] == 0, toronto_merged.columns[[2] + list(range(6, toronto_merged.shape[1]))]]


# In[56]:


# Cluster 2 (1) - Health places to visit around.
toronto_merged.loc[toronto_merged["Cluster Labels"] == 1, toronto_merged.columns[[2] + list(range(6, toronto_merged.shape[1]))]]


# In[57]:


# Cluster 3 (2) - Places to spend time.
toronto_merged.loc[toronto_merged["Cluster Labels"] == 2, toronto_merged.columns[[2] + list(range(6, toronto_merged.shape[1]))]]


# In[58]:


# Cluster 4 (3) - Parks.
toronto_merged.loc[toronto_merged["Cluster Labels"] == 3, toronto_merged.columns[[2] + list(range(6, toronto_merged.shape[1]))]]


# In[59]:


# Cluster 5 (4) - Coffee shops.
toronto_merged.loc[toronto_merged["Cluster Labels"] == 4, toronto_merged.columns[[2] + list(range(6, toronto_merged.shape[1]))]]


# In[60]:


# Cluster 6 (5) - Acessories store.
toronto_merged.loc[toronto_merged["Cluster Labels"] == 5, toronto_merged.columns[[2] + list(range(6, toronto_merged.shape[1]))]]


# In[61]:


# Cluster 7 (6) - Sushi restaurants
toronto_merged.loc[toronto_merged["Cluster Labels"] == 6, toronto_merged.columns[[2] + list(range(6, toronto_merged.shape[1]))]]


# In[62]:


# Cluster 8 (7) - Lakes.
toronto_merged.loc[toronto_merged["Cluster Labels"] == 7, toronto_merged.columns[[2] + list(range(6, toronto_merged.shape[1]))]]


# In[ ]:




