import torch
import torchvision.transforms as T


def convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transforms(tencrop=False):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    train_transform = T.Compose([
        T.RandomResizedCrop(size=224, scale=(0.6, 1.0), ratio=(0.999, 1.001), antialias=True),
        convert_image_to_rgb,
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.Normalize(mean, std),
    ])

    if tencrop:
        eval_transform = T.Compose([
            T.Resize(256, antialias=True),
            convert_image_to_rgb,
            T.TenCrop(224),
            T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
            T.Lambda(lambda crops: torch.stack([T.Normalize(mean, std)(crop) for crop in crops])),
        ])
    else:
        eval_transform = T.Compose([
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    return train_transform, eval_transform