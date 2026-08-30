from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(resolution=224, rotation=15, crop_pad=4):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation),
        transforms.RandomCrop(resolution, padding=crop_pad),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform(resolution=224):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_tta_transform(resolution=224):
    return transforms.Compose([
        transforms.Resize((int(resolution * 1.14), int(resolution * 1.14))),
        transforms.TenCrop(resolution),
        transforms.Lambda(lambda crops: [transforms.ToTensor()(c) for c in crops]),
        transforms.Lambda(lambda tensors: [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(t) for t in tensors]),
    ])
