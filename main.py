import logging
import argparse
import torch 
import torch.nn as nn
from dataloader import get_dataloaders
from trainer import Trainer
from utils import set_logger
import models

def get_model(model_name, args):
    assert(model_name in ["cnn", "rnn", "lstm", "efficientnet"])
    if (model_name == "cnn"): return models.CNNModel(5)
    elif (model_name == "rnn"): return models.RNNModel(input_size=1, hidden_size=64, num_layers=3)
    elif (model_name == "lstm"): return models.LSTMModel(input_size=1, hidden_size=64, num_layers=3)
    elif (model_name == "efficientnet"): return models.EfficientNetModel(5)
    else: return nn.Linear(187, 10)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(args.train_path, args.test_path, args.batch_size, args.image_transform)

    # Initialize model, optimizer, loss
    model = get_model(args.model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    # Training
    trainer = Trainer(model, device, loss_func, optimizer)
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    for e in range(args.n_epochs):
        loss_t, acc_t = trainer.train(train_loader)
        loss_v, acc_v = trainer.validate(val_loader)
        train_loss.append(loss_t)
        train_acc.append(acc_t)
        val_loss.append(loss_v)
        val_acc.append(acc_v)
        logging.info(f"Epoch {e} | loss_t:{loss_t:.5f} acc_t:{acc_t:.5f} | loss_v:{loss_v:.5f} acc_v:{acc_v:.5f}")


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="rnn", choices=["cnn", "rnn", "lstm", "efficientnet"])
    parser.add_argument("--image_transform", type=str, default="none", choices=["none", "gaf", "mtf"])
    parser.add_argument("--train_path", type=str, default="data/mitbih_train.csv")
    parser.add_argument("--test_path", type=str, default="data/mitbih_test.csv")
    parser.add_argument("--logging", action='store_true')    
    args = parser.parse_args()
    set_logger(args.logging)    
    logging.info(args)

    main(args)