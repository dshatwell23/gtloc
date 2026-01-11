import math

import torch
import torch.nn as nn

from .rff import GaussianEncoding


# Constants
MINUTE_TO_HOUR  = 1 / 60
SECOND_TO_HOUR  = MINUTE_TO_HOUR / 60

DAY_TO_MONTH    = 1 / (365 / 12)
HOUR_TO_MONTH   = DAY_TO_MONTH / 24
MINUTE_TO_MONTH = MINUTE_TO_HOUR * HOUR_TO_MONTH
SECOND_TO_MONTH = SECOND_TO_HOUR * HOUR_TO_MONTH


def angular_time_representation(T):
    month, day, hour, minute, second = torch.chunk(T.float(), 5, dim=1)

    # Convert integer local datetime to month and day with decimal places
    month_d = (month - 1) + (day - 1) * DAY_TO_MONTH
    hour_d = hour + minute * MINUTE_TO_HOUR + second * SECOND_TO_HOUR

    # Transform month and day to angles between [-pi, +pi)
    theta = 2 * math.pi * month_d / 12 - math.pi
    phi = 2 * math.pi * hour_d / 24 - math.pi

    return torch.cat((theta, phi), dim=1)


class TimeEncoderCapsule(nn.Module):
    def __init__(self, sigma, embedding_dim=512, dropout_prob=None):
        super(TimeEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=embedding_dim//2)
        self.sigma = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(embedding_dim, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob) if dropout_prob else nn.Identity(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob) if dropout_prob else nn.Identity(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob) if dropout_prob else nn.Identity())
        self.head = nn.Sequential(nn.Linear(1024, embedding_dim))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class TimeEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], embedding_dim=512, dropout_prob=None):
        super(TimeEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.embedding_dim = embedding_dim

        for i, s in enumerate(self.sigma):
            self.add_module('TimeEnc' + str(i), TimeEncoderCapsule(sigma=s, embedding_dim=embedding_dim, dropout_prob=dropout_prob))

    def forward(self, time):
        time = angular_time_representation(time)
        time_features = torch.zeros(time.shape[0], self.embedding_dim, device=time.device)

        for i in range(self.n):
            time_features += self._modules['TimeEnc' + str(i)](time)
        
        return time_features