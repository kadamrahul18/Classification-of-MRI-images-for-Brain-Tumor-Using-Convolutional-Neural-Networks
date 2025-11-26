import albumentations as A


def get_training_augmentation():
    """Return albumentations Compose for training images and masks."""
    return A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(0, 90), p=0.5),
            A.ShiftScaleRotate(shift_limit=(0, 0.1), rotate_limit=(0, 0), scale_limit=(0, 0), p=0.5),
            A.Transpose(p=0.5),
        ], p=1),
    ])
