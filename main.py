"""A pytorch implementation of transformer model."""
import argparse
import json

from train import train_model


def get_parser():
    """Gets parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        default="./configs/config.json",
                        help="config file path.")
    parser.add_argument("-t", "--training", default=True,
                        help="Training mode if True inference mode if False.")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)

    print(config)

    train_model()


if __name__ == '__main__':
    main()

