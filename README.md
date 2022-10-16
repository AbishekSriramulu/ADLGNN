
# ADLGNN
This is a PyTorch implementation of the paper: 

**Adaptive Dependency Learning Graph Neural Network for Multivariate Forecasting.** (Under review)



## Environment setup

#### Version:

**python:** 3.7.6
**pip:** 21.x

The model is implemented using Python3 with dependencies specified in `requirements.txt`

- Install packages using below command:

```
pip install requirements.txt
```

## Data Preparation
### Multivariate time series datasets

European Electricity load data is originally from https://zenodo.org/record/999150

or download the preprocessed data from https://drive.google.com/drive/folders/1QyM_QsscRMld1fhVPKMge3JSOfzrcTVn?usp=sharing

unzip & copy it to /data folder.  

## Model Training and Validation

* European Electricity load data

```
python train_single_step.py --save ./model-RE12.pt  --pretrained_model ./model-RE6.pt --data ./data/og_dataset.csv --num_nodes 1494 --batch_size 12 --epochs 30 --horizon 24 --predefinedA_path ./data/og_recon_adj.csv

```

