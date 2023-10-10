import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 100)
pd.set_option('float_format', '{:,.2f}'.format)

from sklearn.cluster import KMeans
import sklearn.metrics as metrics

import warnings
warnings.filterwarnings("ignore")

import os
import seaborn as sns

# import subprocess

# # Use subprocess to run the command
# command = ['wget', '-q', 'https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf']

# # Use subprocess.run()
# result = subprocess.run(command, capture_output=True, text=True)

# import matplotlib as mpl
# mpl.font_manager.fontManager.addfont('thsarabunnew-webfont.ttf')
# mpl.rc('font', family='TH Sarabun New')

# # download data
# !gdown 1hoUo5M2gKzuJ1t1sztenJKqgcY2nPclc


def cleanColumnName(df):
    ## Clean column names
    df.columns = df.columns.str.strip().str.lower()\
                    .str.replace(" ","_")\
                    .str.replace("-","_")\
                    .str.replace("\/","_")\
                    .str.replace(".","")
    return df

def normalizeStandard(df):
    return (df-df.mean())/df.std()

def main(dataset_path, output_dir, drop_cols):
    # df = pd.read_csv('..\datasets\Mall_Customers.csv')
    df = pd.read_csv(dataset_path)
    df = cleanColumnName(df)
    df_kmean = df.copy().drop(columns=drop_cols).set_index('customerid')
    df_kmean_norm = normalizeStandard(df_kmean)

    scores = {}
    for i in range(2,13):
        labels = KMeans(n_clusters=i, random_state=42).fit(df_kmean_norm).labels_
        score = metrics.silhouette_score(df_kmean_norm,
                                    labels,
                                    random_state=42)
        scores[i] = score
    
    selected_n_cluster = max(scores, key=scores.get)

    km = KMeans(n_clusters=selected_n_cluster, random_state=42).fit_predict(df_kmean_norm)
    df_kmean['cluster'] = km

    # display((df_kmean
    #         .groupby('cluster').mean())

    #         # count each cluster
    #         .join(df_kmean
    #         ['cluster']
    #         .value_counts().to_frame('cluster_count'))
    #         .sort_values(by=['cluster_count'], ascending=False)

    #         # heatmap
    #         .style.background_gradient(cmap='YlGn')
    #         )

    os.makedirs(output_dir, exist_ok=True)
    # fig = df_kmean.boxplot(by='cluster', return_type='axes')
    
    df_kmean['cluster'].value_counts().plot.bar()
    
    for col in df_kmean.columns:
        if col != 'cluster':
            plot = df_kmean[[col, 'cluster']].boxplot(by='cluster')
            fig = plot.get_figure()
            fig.savefig(f"{output_dir}\\boxplot_{col}.png")
    
    df_kmean.to_csv(f"{output_dir}\\df_kmean.csv", index=False)
    return df_kmean

if __name__ == "__main__":
    main(dataset_path="..\datasets\Mall_Customers.csv",
         output_dir="output",
         drop_cols=['gender'])