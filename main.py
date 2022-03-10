import logging
import argparse
import torch 
import torch.nn as nn
from dataloader import get_dataloaders
from trainer import Trainer
from utils import set_logger

def get_model(model_name, args):
    assert(model_name in ["cnn", "rnn", "lstm"])
    if (model_name == "cnn"): return None
    elif (model_name == "rnn"): return nn.Linear(187, 10)
    elif (model_name == "lstm"): return None
    else: return nn.Linear(187, 10)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(args.train_path, args.test_path, args.batch_size)

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
    set_logger()    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, default="rnn")
    parser.add_argument("--train_path", type=str, default="data/mitbih_train.csv")
    parser.add_argument("--test_path", type=str, default="data/mitbih_test.csv")
    args = parser.parse_args()
    logging.info(args)

    main(args)