import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import datetime
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))

# simulate ride data function 

def sim_ride_distances():
    logging.info("Simulating ride distances...")
    ride_dists = np.concatenate(
        (
            10 * np.random.random(size=370),
            30 * np.random.random(size=10), #long dist
            10 * np.random.random(size=10), #same dist
            10 * np.random.random(size=10) #same dist
        )
    )
    return ride_dists

# simulate ride speeds 
def sim_ride_speeds():
    logging.info('simualting ride speeds')
    ride_speeds = np.concatenate(
        (
            np.random.normal(loc=30, scale=5, size=370),
            np.random.normal(loc=30, scale=5, size=10), #same speed
            np.random.normal(loc=50, scale=10, size=10), #high speed
            np.random.normal(loc=15, scale=4, size=10) #low speed
        )
    )
    return ride_speeds

# simulate ride data
def sim_ride_data():
    logging.info('simulating ride data')
    ride_dists = sim_ride_distances()
    ride_speeds = sim_ride_speeds()
    ride_times = ride_dists/ride_speeds
    
    df = pd.DataFrame(
        {
            'ride_dist': ride_dists,
            'ride_time': ride_times,
            'ride_speed': ride_speeds
        }
    )
    ride_ids = datetime.datetime.now().strftime("%Y%m%d") + df.index.astype(str)
    df['ride_ids'] = ride_ids
    return df 

# clustering with dbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics

def plt_cluster_results(data, labels, core_samples_mask, n_clusters):
    fig = plt.figure(figsize=(10, 10))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0,0,0,1]
        class_member_mask = (labels == k)
        xy = data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
        xy = data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '^', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    plt.xlabel('Standard scaled ride dist.')
    plt.ylabel('Standard scaled ride time')
    plt.title('estmated num of clusters: %d' % n_clusters)
    plt.savefig('taxi-rides.png')
