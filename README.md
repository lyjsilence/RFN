# RFN (Recurrent Flow Networks)
Probabilistic Learning of Multivariate Time Series with Temporal Irregularity

### Requirements
Python == 3.8.   
Pytorch: 1.8.1+cu102, torchdiffeq: 0.2.2, Sklearn:0.23.2, Numpy: 1.19.2, Pandas: 1.1.3, Matplotlib: 3.3.2   
All the codes are run on GPUs by default. 

### Correlated Geometric Brownian Motion (Corr-GBM) Experiments

Train the Corr-GBM synchronous dataset
```
python3 Corr_GBM.py --type sync --model_name RFN --marginal GRUODE --num_exp 5 
```

Train the Corr-GBM asynchronous dataset
```
python3 Corr_GBM.py --type async --model_name RFN --marginal GRUODE --num_exp 5 
```
The marginal can be replaced by GRU-D, ODERNN, and ODELSTM


### Physical Activities (MuJoCo) Experiments

To generate Physical Activities (MuJoCo) dataset, the dm_control module needs to be installed in advance.


Train the MuJoCo synchronous dataset
```
python3 MuJoCo.py --type sync --model_name RFN --marginal GRUODE --num_exp 5 
```

Train the MuJoCo asynchronous dataset
```
python3 MuJoCo.py --type async --model_name RFN --marginal GRUODE --num_exp 5 
```
The marginal can be replaced by GRU-D, ODERNN, and ODELSTM


### Climate Records (USHCN) Experiments
USHCN dataset can be downloaded and preprocessed from GRU-ODE-Bayes
https://github.com/edebrouwer/gru_ode_bayes, or the data can be found in data/USHCN folder.


Train the USHCN experiment
```
python3 Climate.py -model_name RFN --marginal GRUODE --num_exp 5 
```

### Financial Transactions (NASDAQ 100) Experiments
NASDAQ 100 dataset can be downloaded from 
http://cseweb.ucsd.edu/∼yaq007/NASDAQ100 stock data.html, or the data can be found in data/NASDAQ folder.

Train the NASDAQ experiment
```
python3 NASDAQ.py -model_name RFN --marginal GRUODE --num_exp 5 
```

