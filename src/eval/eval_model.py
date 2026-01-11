import math

import torch
import torch.nn.functional as F

from tqdm import tqdm


def encode_gallery(encoder, gallery, chunk_size=4096):
    features = []
    num_samples = gallery.size(0)

    encoder.eval()
    with torch.inference_mode():
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            chunk = gallery[start_idx:end_idx]  # no .to() here
            chunk_features = F.normalize(encoder(chunk), dim=1)
            features.append(chunk_features)

    return torch.cat(features, dim=0)


def haversine(gps1, gps2):
    R = 6371.0  # Earth radius in kilometers

    lat1 = torch.deg2rad(gps1[:, 0])
    lon1 = torch.deg2rad(gps1[:, 1])
    lat2 = torch.deg2rad(gps2[:, 0])
    lon2 = torch.deg2rad(gps2[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance = R * c
    return distance


def compute_time_errors(ref_time, pred_time):
    ref_month = ref_time[:, 0] + (ref_time[:, 1] - 1) / 31.0
    ref_hour = ref_time[:, 2] + ref_time[:, 3] / 60.0 + ref_time[:, 4] / 3600.0

    pred_month = pred_time[:, 0] + (pred_time[:, 1] - 1) / 31.0
    pred_hour = pred_time[:, 2] + pred_time[:, 3] / 60.0 + pred_time[:, 4] / 3600.0

    month_errors = torch.abs(ref_month - pred_month)
    month_errors = torch.min(month_errors, 12 - month_errors)

    hour_errors = torch.abs(ref_hour - pred_hour)
    hour_errors = torch.min(hour_errors, 24 - hour_errors)

    tps = 1 - torch.sqrt((month_errors / 6.0)**2 + (hour_errors / 12.0)**2) / math.sqrt(2)

    return month_errors, hour_errors, tps


def chunked_argmax_gallery_chunked(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    *,
    tencrop: bool = False,
    gallery_chunk_size: int = 8192,
) -> torch.Tensor:
    assert gallery_features.device == query_features.device, \
        f"Device mismatch: query {query_features.device} vs gallery {gallery_features.device}"

    device = query_features.device
    G = gallery_features.size(0)

    if not tencrop:
        # query: [B, D]
        B, D = query_features.shape
        best_scores = torch.full((B,), -float("inf"), device=device)
        best_indices = torch.zeros((B,), dtype=torch.long, device=device)

        for g0 in range(0, G, gallery_chunk_size):
            g1 = min(g0 + gallery_chunk_size, G)
            g = gallery_features[g0:g1]            # [g, D]
            sims = query_features @ g.t()          # [B, g]
            scores, idx = sims.max(dim=1)          # [B]
            better = scores > best_scores
            best_scores[better] = scores[better]
            best_indices[better] = g0 + idx[better]

        return best_indices.cpu()

    else:
        # query: [B, n, D]
        B, n, D = query_features.shape
        q_flat = query_features.reshape(B * n, D)  # [B*n, D]

        best_scores = torch.full((B,), -float("inf"), device=device)
        best_indices = torch.zeros((B,), dtype=torch.long, device=device)

        for g0 in range(0, G, gallery_chunk_size):
            g1 = min(g0 + gallery_chunk_size, G)
            g = gallery_features[g0:g1]            # [g, D]

            sims = q_flat @ g.t()                  # [B*n, g]
            sims = sims.view(B, n, -1).mean(dim=1) # [B, g] mean over crops

            scores, idx = sims.max(dim=1)          # [B]
            better = scores > best_scores
            best_scores[better] = scores[better]
            best_indices[better] = g0 + idx[better]

        return best_indices.cpu()


@torch.inference_mode()
def eval_model(model, geo_loaders, time_loaders, tencrop=False, device="cuda"):
    model.eval()
    device = torch.device(device)

    ##############################################
    # EVAL GEO-LOCATION
    ##############################################
    print("Evaluating geolocation...")

    geo_results = {}
    geo_thresholds = [1, 25, 200, 750, 2500]

    for name, loader in geo_loaders.items():
        # # Keep raw gallery coords on CPU to avoid holding coords + features on GPU
        # if name == "im2gps3k":
        #     gps_gallery_cpu = model.gps_gallery_100k  # CPU tensor [G, 2]
        # elif name == "gws15k":
        #     gps_gallery_cpu = model.gps_gallery_500k  # CPU tensor [G, 2]
        # else:
        #     raise ValueError(f"Unknown geo dataset: {name}")
        
        gps_gallery_cpu = model.gps_gallery_100k

        # Encode gallery on GPU (coords -> embedding). Then free coords GPU copy.
        gps_gallery_dev = gps_gallery_cpu.to(device, non_blocking=True)
        gps_gallery_features = encode_gallery(model.location_encoder, gps_gallery_dev, chunk_size=65536)
        del gps_gallery_dev  # free raw coords from GPU ASAP

        # Streaming accumulators (CPU)
        all_errors = []
        acc_counts = {thr: 0 for thr in geo_thresholds}
        total = 0

        for imgs, metadata in tqdm(loader, desc=f"{name}"):
            imgs = imgs.to(device, non_blocking=True)

            if tencrop:
                b, n_crops, c, h, w = imgs.size()
                imgs = imgs.reshape(-1, c, h, w)

            feats = F.normalize(model.image_encoder(imgs), dim=1)

            if tencrop:
                feats = feats.view(b, n_crops, -1)

            # Retrieve indices for this batch (exact, gallery-chunked)
            pred_idx = chunked_argmax_gallery_chunked(
                feats, gps_gallery_features, tencrop=tencrop, gallery_chunk_size=16384
            )  # CPU LongTensor [B]

            # Compute errors on CPU
            pred_gps = gps_gallery_cpu[pred_idx]          # CPU [B, 2]
            ref_gps = metadata["gps"].cpu()               # CPU [B, 2]
            errors = haversine(ref_gps, pred_gps).cpu()   # CPU [B]

            all_errors.append(errors)
            bs = errors.numel()
            total += bs
            for thr in geo_thresholds:
                acc_counts[thr] += (errors <= thr).sum().item()

            # Drop batch tensors promptly
            del imgs, feats, pred_idx, pred_gps, ref_gps, errors

        location_errors = torch.cat(all_errors, dim=0)
        median_error = torch.median(location_errors).item()

        geo_results[name] = {
            "median_error_km": median_error,
            **{f"acc@{thr}km": acc_counts[thr] / max(total, 1) for thr in geo_thresholds},
        }

        # Free the big gallery embedding
        del gps_gallery_features, all_errors, location_errors

    ##############################################
    # EVAL TIME
    ##############################################
    print("Evaluating time...")

    # Keep raw time gallery on CPU, encode features on GPU once
    time_gallery_cpu = model.time_gallery
    time_gallery_dev = time_gallery_cpu.to(device, non_blocking=True)
    time_gallery_features = encode_gallery(model.time_encoder, time_gallery_dev, chunk_size=65536)
    del time_gallery_dev

    time_results = {}
    for name, loader in time_loaders.items():
        # Streaming sums (CPU scalars)
        sum_month = 0.0
        sum_hour = 0.0
        sum_tps = 0.0
        total = 0

        for imgs, metadata in tqdm(loader, desc=f"{name}"):
            imgs = imgs.to(device, non_blocking=True)

            if tencrop:
                b, n_crops, c, h, w = imgs.size()
                imgs = imgs.reshape(-1, c, h, w)

            feats = F.normalize(model.image_encoder(imgs), dim=1)

            if tencrop:
                feats = feats.view(b, n_crops, -1)

            pred_idx = chunked_argmax_gallery_chunked(
                feats, time_gallery_features, tencrop=tencrop, gallery_chunk_size=16384
            )  # CPU [B]

            pred_time = time_gallery_cpu[pred_idx].cpu()     # CPU [B, 5]
            ref_time = metadata["time"].cpu()                # CPU [B, 5]

            month_err, hour_err, tps = compute_time_errors(ref_time, pred_time)
            bs = month_err.numel()
            total += bs

            sum_month += month_err.sum().item()
            sum_hour += hour_err.sum().item()
            sum_tps += tps.sum().item()

            del imgs, feats, pred_idx, pred_time, ref_time, month_err, hour_err, tps

        time_results[name] = {
            "mean_month_error": sum_month / max(total, 1),
            "mean_hour_error": sum_hour / max(total, 1),
            "mean_tps": sum_tps / max(total, 1),
        }

    del time_gallery_features

    return {"geo": geo_results, "time": time_results}
