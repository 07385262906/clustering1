import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN
from scipy.optimize import curve_fit

df = pd.read_csv('WDI_Dataset.csv', skiprows=3)

df.head()

df = df.drop('Unnamed: 65', axis=1)

df_melt = df.melt(id_vars=["Country Name","Country Code","Indicator Name","Indicator Code"], var_name="Year", value_name="Value")

df_melt['Year'] = pd.to_datetime(df_melt['Year'])
df_melt.set_index('Year', inplace=True)

df_melt.groupby('Indicator Name')

df_pop = df_melt.groupby('Indicator Name').get_group('Population growth (annual %)')

df_pop_usa = df_pop[df_pop['Country Name'] == 'United States']
df_pop_usa = df_pop_usa.groupby('Year')['Value'].sum()

df_pop_brazil = df_pop[df_pop['Country Name'] == 'Brazil']
df_pop_brazil = df_pop_brazil.groupby('Year')['Value'].sum()

df_pop_afghanistan = df_pop[df_pop['Country Name'] == 'Afghanistan']
df_pop_afghanistan = df_pop_afghanistan.groupby('Year')['Value'].sum()

plt.plot(df_pop_usa.index, df_pop_usa, label='USA')
plt.plot(df_pop_brazil.index, df_pop_brazil, label='Brazil')
plt.plot(df_pop_afghanistan.index, df_pop_afghanistan, label='Afghanistan')

plt.xlabel('Year')
plt.ylabel('Population growth (annual %)')
plt.title('Population growth of Country over Time')
plt.legend()
plt.show()

df_pop = df_melt.groupby('Indicator Name').get_group('Population, total')

df_pop_usa = df_pop[df_pop['Country Name'] == 'United States']
df_pop_usa = df_pop_usa.groupby('Year')['Value'].sum()

df_pop_brazil = df_pop[df_pop['Country Name'] == 'Brazil']
df_pop_brazil = df_pop_brazil.groupby('Year')['Value'].sum()

df_pop_afghanistan = df_pop[df_pop['Country Name'] == 'Afghanistan']
df_pop_afghanistan = df_pop_afghanistan.groupby('Year')['Value'].sum()

plt.plot(df_pop_usa.index, df_pop_usa, label='USA')
plt.plot(df_pop_brazil.index, df_pop_brazil, label='Brazil')
plt.plot(df_pop_afghanistan.index, df_pop_afghanistan, label='Afghanistan')

plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population of Country over Time')
plt.legend()
plt.show()

X = df_pop_usa.values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_kmeans = kmeans.predict(X)

# Add cluster labels to the dataset
df_pop_usa['cluster'] = y_kmeans

# Plot the clusters and the cluster centers
plt.scatter(X, y_kmeans, c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], np.arange(3), c='black', s=200, alpha=0.5)
plt.title('Population of USA by Clusters')
plt.xlabel('Population')
plt.ylabel('Cluster')
plt.show()

X = df_pop_brazil.values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_kmeans = kmeans.predict(X)

# Add cluster labels to the dataset
df_pop_brazil['cluster'] = y_kmeans

# Plot the clusters and the cluster centers
plt.scatter(X, y_kmeans, c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], np.arange(3), c='black', s=200, alpha=0.5)
plt.title('Population of Brazil by Clusters')
plt.xlabel('Population')
plt.ylabel('Cluster')
plt.show()

X = df_pop_afghanistan.values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_kmeans = kmeans.predict(X)

# Add cluster labels to the dataset
df_pop_afghanistan['cluster'] = y_kmeans

# Plot the clusters and the cluster centers
plt.scatter(X, y_kmeans, c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], np.arange(3), c='black', s=200, alpha=0.5)
plt.title('Population of Afghanistan by Clusters')
plt.xlabel('Population')
plt.ylabel('Cluster')
plt.show()

#defining function to calculate confidence range 
def err_ranges(xdata, ydata, func):
    popt, pcov = curve_fit(func, xdata, ydata)
    perr = np.sqrt(np.diag(pcov))
    upper = func(xdata, *(popt + perr))
    lower = func(xdata, *(popt - perr))
    return upper, lower

#defining exponential growth model
def exp_func(x, a, b):
    return a * np.exp(x * b)

df_pop_usa = df_pop[df_pop['Country Name'] == 'United States']
df_pop_usa = df_pop_usa.groupby('Year')['Value'].sum()

#getting x and y data
xdata = df_pop_usa.index.year
ydata = df_pop_usa.values

#fitting the data with exponential growth model
popt, pcov = curve_fit(exp_func, xdata, ydata)

#calculating the confidence range 
upper, lower = err_ranges(xdata, ydata, exp_func)

#plotting the best fitting function 
plt.plot(xdata, ydata, 'b-', label='data')
plt.plot(xdata, exp_func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

#plotting the confidence range
plt.fill_between(xdata, upper, lower, color='gray', alpha=0.2, label='confidence range')
plt.xlabel('Year')
plt.ylabel('Population, total')
plt.legend()
plt.show()

#predicting population value in 10 and 20 years 
pred_10 = exp_func(10, *popt)
pred_20 = exp_func(20, *popt)

#calculating confidence range for pred_10 and pred_20
upper_10, lower_10 = err_ranges(10, pred_10, exp_func)
upper_20, lower_20 = err_ranges(20, pred_20, exp_func)

print("The predicted population value in 10 years is between", lower_10, "and", upper_10)
print("The predicted population value in 20 years is between", lower_20, "and", upper_20)

