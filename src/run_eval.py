import os
import argparse
import json

import torch
from torch.utils.data import DataLoader

import numpy as np

from utils import print_results
from src.configs import config
from model import GTLoc
from datasets.dataloader import GeoTemporalDataset, retvals
from datasets.transforms import get_transforms
from eval.eval_model import eval_model


def main():

    ##############################################
    # GENERAL SETUP
    ##############################################

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint to evaluate')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to directory containing the evaluation metrics')
    args = parser.parse_args()

    ##############################################
    # SELECT DEVICE
    ##############################################

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device('cpu')
    
    print(f"Device:         {device}")


    ##############################################
    # MODEL
    ##############################################

    model = GTLoc(
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        queue_size=config.queue_size,
        time_sigma=config.time_sigma,
        loc_sigma=config.loc_sigma,
        freeze_backbone=config.freeze_backbone,
        galleries=config.galleries,
        time_dropout=config.dropout_prob,
    )

    state_dict = torch.load(args.ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)


    ##############################################
    # DATALOADERS
    ##############################################

    print("Creating dataloaders...")

    _, val_transform = get_transforms(tencrop=config.tencrop)

    im2gps3k_dataset = GeoTemporalDataset(
        metadata_path=config.im2gps3k_metadata_path,
        imgs_path=config.im2gps3k_imgs_path,
        transform=val_transform,
        fields=retvals.GEO,
        seed=config.seed,
    )

    gws15k_dataset = GeoTemporalDataset(
        metadata_path=config.gws15k_metadata_path,
        imgs_path=config.gws15k_imgs_path,
        transform=val_transform,
        fields=retvals.GEO,
        seed=config.seed,
    )

    skyfinder_dataset = GeoTemporalDataset(
        metadata_path=config.skyfinder_val_metadata_path,
        imgs_path=config.skyfinder_imgs_path,
        transform=val_transform,
        fields=retvals.GEOTEMP,
        seed=config.seed,
    )

    eval_bsz = max(config.eval_bsz // 10, 1) if config.tencrop else config.eval_bsz

    geo_loaders = {
        'gws15k': DataLoader(gws15k_dataset, batch_size=eval_bsz, num_workers=config.num_workers),
        'im2gps3k': DataLoader(im2gps3k_dataset, batch_size=eval_bsz, num_workers=config.num_workers),
    }

    time_loaders = {
        'skyfinder': DataLoader(skyfinder_dataset, batch_size=eval_bsz, num_workers=config.num_workers),
    }


    ##############################################
    # EVALUATION
    ##############################################

    results = eval_model(model, geo_loaders, time_loaders, tencrop=config.tencrop, device=device)
    print_results(results)

    # Save results to JSON
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation results to {results_path}")


if __name__ == '__main__':
    main()