import math

import torch
import torch.nn as nn

import numpy as np

from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .time_encoder import TimeEncoder


class GTLoc(nn.Module):
    def __init__(
            self,
            hidden_dim=768,
            embedding_dim=512,
            queue_size=4096,
            time_sigma=[2**0, 2**4, 2**8],
            loc_sigma=[2**0, 2**4, 2**8],
            galleries='data_dist',
            freeze_backbone=True,
            temperature=None,
            time_dropout=None,
        ):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.time_sigma = time_sigma
        self.loc_sigma = loc_sigma
        self.freeze_backbone = freeze_backbone
        self.galleries = galleries
        self.temperature = temperature
        self.time_dropout = time_dropout
        
        self.img_gps_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.img_time_logit_scale = temperature or nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if temperature:
            self.img_time_logit_scale = torch.ones([]) * np.log(1 / temperature)
        else:
            self.img_time_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gps_time_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ig2t_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.it2g_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.gt2i_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        toggle_freeze = freeze_backbone
        self.image_encoder = ImageEncoder(hidden_dim, embedding_dim, toggle_freeze)
        
        self.location_encoder = LocationEncoder(loc_sigma, embedding_dim)
        self.time_encoder = TimeEncoder(time_sigma, embedding_dim, dropout_prob=time_dropout)

        self._freeze_modules(freeze_backbone)

        self._load_galleries()

        self._initialize_gps_queue(queue_size)
        self._initialize_time_queue(queue_size)

        self.device = "cpu"

    def _freeze_modules(self, backbone=True):
        if backbone:
            image_encoder_params = self.image_encoder.backbone.parameters()
            for param in image_encoder_params:
                param.requires_grad = False
    
    def _load_galleries(self):            
        self.gps_gallery_100k = torch.tensor(np.load(f'model/gps_galleries/mp16.npy'), dtype=torch.float32)
        self.gps_gallery_500k = torch.tensor(np.load(f'model/gps_galleries/mp16_500k.npy'), dtype=torch.float32)
        if self.galleries == 'data_dist':
            self.time_gallery = torch.tensor(np.load('model/time_galleries/cvt.npy'), dtype=torch.float32)
        else: # random
            N = 500_000
            M = math.ceil(math.sqrt(N))

            # 2) Sample days of year and times of day
            days = torch.linspace(1, 365, M).round().to(torch.int64)                       # shape (M,)
            seconds_per_day = 24 * 3600
            times = torch.linspace(0, seconds_per_day - 1, M).round().to(torch.int64)      # shape (M,)

            # 3) Build full grid of size M*M
            day_grid  = days.repeat_interleave(M)   # (M*M,)
            time_grid = times.repeat(M)             # (M*M,)

            # 4) Convert day_of_year → (month, day_of_month)
            month_lengths = torch.tensor([31,28,31,30,31,30,31,31,30,31,30,31], dtype=torch.int64)
            month_edges   = torch.cumsum(month_lengths, dim=0)  # [31, 59, 90, …, 365]
            month         = torch.bucketize(day_grid, month_edges) + 1
            prev_edges    = torch.cat((torch.tensor([0], dtype=torch.int64), month_edges[:-1]))
            day           = day_grid - prev_edges[month - 1]

            # 5) Convert seconds → (hour, minute, second)
            hour   = time_grid // 3600
            minute = (time_grid % 3600) // 60
            second = time_grid % 60

            # 6) Stack into (M*M, 5)
            full = torch.stack([month, day, hour, minute, second], dim=1)

            # 7) If we have more than N rows, randomly pick N without replacement
            total = full.size(0)
            if total > N:
                perm = torch.randperm(total)[:N]
                result = full[perm]
            else:
                result = full
            self.time_gallery = result.float()

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.gps_queue *= torch.tensor([[90], [180]])
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    def _initialize_time_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("time_queue", torch.randn(5, self.queue_size))
        self.time_queue = nn.functional.normalize(self.time_queue, dim=0)
        self.time_queue = torch.floor(self.time_queue * torch.tensor([[12], [30], [24], [60], [60]])) + torch.tensor([[1], [1], [0], [0], [0]])
        self.register_buffer("time_queue_ptr", torch.zeros(1, dtype=torch.long))

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.time_encoder.to(device)
        self.img_gps_logit_scale.data = self.img_gps_logit_scale.data.to(device)
        if self.temperature is None:
            self.img_time_logit_scale.data = self.img_time_logit_scale.data.to(device)
        self.gps_time_logit_scale.data = self.gps_time_logit_scale.data.to(device)
        self.ig2t_logit_scale.data = self.ig2t_logit_scale.data.to(device)
        self.it2g_logit_scale.data = self.it2g_logit_scale.data.to(device)
        self.gt2i_logit_scale.data = self.gt2i_logit_scale.data.to(device)
        return super().to(device)
    
    @torch.no_grad()
    def dequeue_and_enqueue_gps(self, gps):
        """ Update GPS queue
        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Check if the enqueue operation will exceed the bounds of the queue
        if gps_ptr + gps_batch_size > self.queue_size:
            # Amount to fill till the end of the queue
            space_till_end = self.queue_size - gps_ptr
            # Fill till the end of the queue
            self.gps_queue[:, gps_ptr:] = gps.t()[:, :space_till_end]
            # Fill the remaining from the beginning of the queue
            self.gps_queue[:, 0:gps_batch_size - space_till_end] = gps.t()[:, space_till_end:]
        else:
            # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
            self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        
        # Move pointer, handling the wrap-around by modulo operation with queue size
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size
        self.gps_queue_ptr[0] = gps_ptr

    @torch.no_grad()
    def dequeue_and_enqueue_time(self, time):
        """ Update time queue
        Args:
            time (torch.Tensor): time tensor of shape (batch_size, 5)
        """
        time_batch_size = time.shape[0]
        time_ptr = int(self.time_queue_ptr)
        
        assert self.queue_size % time_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {time_batch_size}"

        # Check if the enqueue operation will exceed the bounds of the queue
        if time_ptr + time_batch_size > self.queue_size:
            # Amount to fill till the end of the queue
            space_till_end = self.queue_size - time_ptr
            # Fill till the end of the queue
            self.time_queue[:, time_ptr:] = time.t()[:, :space_till_end]
            # Fill the remaining from the beginning of the queue
            self.time_queue[:, 0:time_batch_size - space_till_end] = time.t()[:, space_till_end:]
        else:
            # Replace the time from ptr to ptr+time_batch_size (dequeue and enqueue)
            self.time_queue[:, time_ptr:time_ptr + time_batch_size] = time.t()
        
        # Move pointer, handling the wrap-around by modulo operation with queue size
        time_ptr = (time_ptr + time_batch_size) % self.queue_size
        self.time_queue_ptr[0] = time_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
    
    def get_time_queue(self):
        return self.time_queue.t()
    
    def forward(self, image, time, location):
        image_features = self.image_encoder(image)
        time_features = self.time_encoder(time)
        location_features = self.location_encoder(location)
        return image_features, time_features, location_features