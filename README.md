# ecg-time-series

## Quick Setup
Download data from Kaggle [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/shayanfazeli/heartbeat) and place it in `./data`.

~~Train+evaluate the model by calling~~
~~python main.py --model (model_name) --image_transform (type_of_transform)~~

Training and evaluation code can be found in `notebooks/rnn_gru_lstm_cleanedup.ipynb`.

For the dynamic time warping and shapelets, they are in `notebooks/linmdtw-cleanedup.ipynb`, `soft_dtw-cleanedup.ipynb`, and `notebooks/shapeled-Cleanedup.ipynb`.

For the transformer models, they are in `python main-trainsformer.py`.
