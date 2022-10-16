import numpy as np
import networkx as nx
import netrd
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped


recons = {
    'CorrelationMatrix':            netrd.reconstruction.CorrelationMatrix(),
    'CorrelationSpanningTree': netrd.reconstruction.CorrelationSpanningTree(),
    'PartialCorrelationInfluence': netrd.reconstruction.PartialCorrelationInfluence(),
    #'PartialCorrelationMatrix': netrd.reconstruction.PartialCorrelationMatrix(),
    }



@background  
def your_function(argument):
    df = pd.read_csv("data/" + path, header = None)
    df = df[-int(len(df)*.1):]
    print()
    print()
    print("working on dataset " + path)
    # kwargs = {'threshold_type':'quantile', 'quantile':0.9}
    # loop over all the reconstruction techniques
    for ri, R1 in list(recons.items()):
        print( "**** starting - " + str(ri) + " ********")
        try:
            R1.fit(np.array(df.T))
            print( "**** Fit successful - " + str(ri) + " ********")
            try:
                adj = pd.DataFrame(R1.results['thresholded_matrix']).abs()
            except:
                adj = pd.DataFrame(R1.results['distance_matrix']).abs()
            adj.replace([np.inf, np.nan], 0.0, inplace=True)
            adj.mask(adj.rank(axis=0, method='min', ascending=False) > 5, 0, inplace = True)  
            adj.mask(adj.rank(axis=1, method='min', ascending=False) > 5, 0, inplace = True) 
            adj.to_csv(path.split(".")[0]+"_"+ri+'.csv', index=False)
            print( "**** file saved ********")
            print()
        except Exception as e:
            print( "**** could not perform - " + str(ri) + " ********")
            print(e)
            print()
            continue
            
# dictionary to store the outputs
datasets= ["traffic.txt.gz", "solar_AL.txt.gz","electricity.txt.gz", "exchange_rate.txt.gz" ]
for path in datasets:
    your_function(path)
    