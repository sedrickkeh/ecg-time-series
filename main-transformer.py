import logging
import argparse
import torch 
import torch.nn as nn
from dataloader import get_dataloaders
from trainer import Trainer
from utils import set_logger
import models
from transformermodule.transformer import Transformer

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(args.train_path, args.test_path, args.batch_size, args.image_transform)

    # Initialize model, optimizer, loss
    EPOCH = 100
    BATCH_SIZE = 3
    LR = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU

    d_model = 512
    d_hidden = 1024
    q = 8
    v = 8
    h = 8           # num heads
    N = 1           # num layers
    dropout = 0.0
    pe = True       # positional encoding
    mask = True  

    d_input = 187
    d_channel = 1
    d_output = 5

    model = Transformer(d_model=d_model, d_input=d_input, d_channel=d_channel, d_output=d_output, d_hidden=d_hidden,
                    q=q, v=v, h=h, N=N, dropout=dropout, pe=pe, mask=mask, device=device).to(device)
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

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", type=str, default="rnn", choices=["cnn", "rnn", "lstm", "efficientnet"])
    parser.add_argument("--image_transform", type=str, default="none", choices=["none", "gaf", "mtf"])
    parser.add_argument("--train_path", type=str, default="data/mitbih_train.csv")
    parser.add_argument("--test_path", type=str, default="data/mitbih_test.csv")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--logging", action='store_true')    
    args = parser.parse_args()
    set_logger(args.logging)    
    logging.info(args)

    main(args)