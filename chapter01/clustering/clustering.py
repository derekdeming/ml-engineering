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

def cluster_label(data, create_show_plot=True):
    data= StandardScaler().fit_transform(data)
    db = DBSCAN(eps=0.3, min_samples=10).fit(data)
    
    # find the labels from clustering
    core_samp_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samp_mask[db.core_sample_indices_] = True 
    labels = db.labels_
    
    # num of clusters in labels (ignoring noise if present)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('estimated num of clusters: %d' % n_clusters_)
    print('estimated num of noise points: %d' %  n_noise_)
    print('silhouette coefficient: %0.3f' % metrics.silhouette_score(data, labels))
    
    run_metadata = {
        'num_clusters': n_clusters_,
        'num_noise_points': n_noise_,
        'silhouette_coefficient': metrics.silhouette_score(data, labels),
        'labels': labels
    }
    if create_show_plot:
        plt_cluster_results(data, labels, core_samp_mask, n_clusters_)
    else: 
        pass
    return run_metadata

if __name__ == "__main__": 
    import os
    file_path = 'taxi-rides.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        logging.info('simulating ride data')
        df = sim_ride_data()
        df.to_csv(file_path, index=False )
        
    # some cool plots 
    plot = df[['ride_dist', 'ride_time']]
    logging.info('clustering and labeling')
    
    results = cluster_label(plot, create_show_plot=True)
    df['label'] = results['labels']
    
    #  output to json 
    logging.info('output to json')
    df.to_json('taxi-rides-labeled.json', orient='records')