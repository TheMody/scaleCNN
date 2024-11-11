from config import *
from eval import eval
from train import train

if __name__ == "__main__":
    train(Baseline=True, pretrained=False, data_is_scaled=True)
    eval(Baseline=True)
    train(Baseline=False, pretrained=True, data_is_scaled=False)