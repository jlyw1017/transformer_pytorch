import numpy as np
import torch
from torchvision.utils import make_grid


class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None,
                 len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.log_step = None

    def _train_epoch(self, epoch):
        """

        Args:
            epoch:

        Returns:

        """
        pass

    def _valid_epoch(self, epoch):
        """

        Args:
            epoch:

        Returns:

        """
        pass


    def _progress(self, batch_idx):
        """

        Args:
            batch_idx:

        Returns:

        """
        pass