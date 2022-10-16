import numpy as np
import networkx as nx
import netrd
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd


recons = {
    'ConvergentCrossMapping':       netrd.reconstruction.ConvergentCrossMapping(),
    'NaiveTransferEntropy': netrd.reconstruction.NaiveTransferEntropy(),
    #'ExactMeanField':               netrd.reconstruction.MeanField(),
    'FreeEnergyMinimization':       netrd.reconstruction.FreeEnergyMinimization(),
    'MarchenkoPastur':              netrd.reconstruction.MarchenkoPastur(),
    'MaximumLikelihoodEstimation':  netrd.reconstruction.MaximumLikelihoodEstimation(),
    'OUInference':                  netrd.reconstruction.OUInference(),
    'ThoulessAndersonPalmer':       netrd.reconstruction.ThoulessAndersonPalmer(),
    }


# dictionary to store the outputs
datasets= ["traffic.txt.gz", "solar_AL.txt.gz","electricity.txt.gz", "exchange_rate.txt.gz" ]
for path in datasets:
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
            adj = pd.DataFrame(R1.results['thresholded_matrix']).abs()
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