import os
import copy
import numpy as np
import torch


def log_results(writer, results, epoch, stage='test'):
    for modality, modality_results in results.items():
        for dataset, dataset_results in modality_results.items():
            for name, value in dataset_results.items():
                writer.add_scalar(f'{stage}/{modality}/{dataset}/{name}', value, epoch) 

def print_results(results):
    print()
    if 'geo' in results:
        for dataset in results['geo']:
            print(f"Geolocation Results for {dataset}:")
            print(f"  Median Error     -> {results['geo'][dataset]['median_error_km']:.4f} km")
            print(f"  Accuracy @1km    -> {100*results['geo'][dataset]['acc@1km']:.4f}%")
            print(f"  Accuracy @25km   -> {100*results['geo'][dataset]['acc@25km']:.4f}%")
            print(f"  Accuracy @200km  -> {100*results['geo'][dataset]['acc@200km']:.4f}%")
            print(f"  Accuracy @750km  -> {100*results['geo'][dataset]['acc@750km']:.4f}%")
            print(f"  Accuracy @2500km -> {100*results['geo'][dataset]['acc@2500km']:.4f}%")
    if 'time' in results:
        for dataset in results['time']:
            print(f"Time Results for {dataset}:")
            print(f"  MAE (months)     -> {results['time'][dataset]['mean_month_error']:.4f} months")
            print(f"  MAE (hours)      -> {results['time'][dataset]['mean_hour_error']:.4f} hours")
