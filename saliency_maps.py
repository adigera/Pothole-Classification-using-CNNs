import copy
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

# from tensorboardX import SummaryWriter
from torchvision import datasets, models, transforms
from skimage import io, exposure
from PIL import Image

# from sklearn.metrics import acc

# plt.ion()  # interactive mode

try:
    plt.switch_backend("qt5agg")
except ImportError:
    pass


def main():

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    model_ft = models.googlenet(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.aux_logits = False
    model_ft.fc = nn.Linear(num_ftrs, 2)
    device = "cpu"
    model_ft = model_ft.to(device)

    # Load model
    model_ft.load_state_dict(torch.load("weightsOld"))
    print("loaded weights")
    
    image = Image.open("data/train/notPotH/negative982.JPG")
    target_class = 1

    prep_image = data_transforms["val"](image)

    print(prep_image.shape)

    prep_image.requires_grad_()
    scores = model_ft(prep_image.reshape(1, *prep_image.shape))

    score_max_index = scores.argmax()
    print("index: ", score_max_index)
    score_max = scores[0, 0]
    score_max.backward()
    print(prep_image.grad.shape)
    saliency = prep_image.grad.numpy()
    saliency = np.abs(saliency)
    saliency = saliency.reshape(224, 224, 3)
    saliency = np.max(saliency, axis=2)
    print(
        f"max: {np.max(saliency):.3f} min: {np.min(saliency):.3f} mean: {np.mean(saliency):.3f}"
    )
    # saliency, _ = torch.max(prep_image.grad.data.abs(), dim=1)

    # print(saliency.shape)
    # saliency = exposure.rescale_intensity(saliency, (0.0, 1.0))
    saliency = exposure.equalize_hist(saliency)

    print(
        f"max: {np.max(saliency):.3f} min: {np.min(saliency):.3f} mean: {np.mean(saliency):.3f}"
    )
    plt.imshow(saliency, cmap=plt.cm.hot)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
