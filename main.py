"""A pytorch implementation of transformer model."""
import argparse

from train import train_model

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data", default="./",
                    help="data source")

parser.add_argument("-o", "--output", default="./result/",
                    help="output path.")

parser.add_argument("-c", "--config", default="./config/",
                    help="config file path.")


parser.add_argument("-t", "--training", default=True,
                    help="Training mode if True inference mode if False.")

def main():
  args = parser.parse_args()
  data_path = args.data
  output_path = args.output
  training_mode = args.training

  train_model()

if __name__ == '__main__':
  main()

