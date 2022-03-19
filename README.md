# ecg-time-series

## Quick Setup
Download data from Kaggle [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/shayanfazeli/heartbeat) and place it in `./data`.

Train+evaluate the model by calling
```bash
python main.py --model (model_name) --image_transform (type_of_transform) 
```

Note: If --image_transform is "gaf" or "mtf", the --model has to be "efficientnet".

Experiments are recorded inside `./logging`