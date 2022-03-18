import logging
import argparse
import pandas as pd
from utils import set_logger
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

def get_model(model_name, args):
    if model_name=="xgboost":
        model = XGBClassifier(learning_rate=args.lr,
                            max_depth=args.max_depth)
    elif model_name=="lightgbm":
        model = LGBMClassifier(learning_rate=args.lr,
                            max_depth=args.max_depth)
    else:
        model = None
    return model


def main(args):
    # Load data
    train, test = pd.read_csv(args.train_path, header=None), pd.read_csv(args.test_path, header=None)
    train_x, train_y = train.loc[:,train.columns!=187], train.loc[:,train.columns==187]
    test_x, test_y = test.loc[:,test.columns!=187], test.loc[:,test.columns==187]

    model = get_model(args.model, args)
    model.fit(train_x,train_y)

    preds = model.predict(test_x)
    accuracy = accuracy_score(test_y, preds)
    print(classification_report(test_y, preds, digits=5))
    logging.info(f"Model: {args.model} | Accuracy: {accuracy:.5f}")


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="xgboost")
    parser.add_argument("--train_path", type=str, default="data/mitbih_train.csv")
    parser.add_argument("--test_path", type=str, default="data/mitbih_test.csv")
    parser.add_argument("--logging", action='store_true')    

    # Model-specific params
    parser.add_argument("--max_depth", type=int, default=4)

    args = parser.parse_args()
    set_logger(args.logging)    
    logging.info(args)

    main(args)