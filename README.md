# Wikipedia Time Series Forecasting with TensorFlow

## DATASET
The full dataset is ~500MB and can be downloaded at : https://www.kaggle.com/c/web-traffic-time-series-forecasting/data

## SOURCE FILES
data_preprocessing.py : script to pre-process the data. Further data preparation (such as one-hot encoding) takes
place within tensorflow.

gru_contextual.py : RNN with GRU cell and contextual features    
rnn_baseline.py : Baseline RNN with no contextual features and simple cell    
rnn_contextual.py : Baseline RNN + contextual features    
seq2seq.py : Encoder Decoder architecture    

utils.py : contains functions used by all models, including preparation of input pipeline for tensorflow     
parse_input_parameters.py : utility script to parse command line args
