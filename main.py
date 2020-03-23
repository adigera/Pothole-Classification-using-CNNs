# The raw code was taken from tutorial on pytorch.org under transfer learning tutorial. Then it was changed as per requirements.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.multiprocessing
import numpy as np
import itertools
import torchvision
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from tensorboardX import SummaryWriter
# from tensorboardX import SummaryWriter
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# from sklearn.metrics import acc

# plt.ion()  # interactive mode

try:
    plt.switch_backend("qt5agg")
except ImportError:
    pass


def main():

    m_Wts_preload = None #"recentPreWeights"        # None/ filename of saved weights
    b_size = 192                                # batch size
    tb_comment = "_1GN_E40_128_Ad_WM_IN_lr-15_b1-0.92_LWts(samSize)_60_epochs"    # comment for tensorboard to save file
    data_dir = "data"                   # data directory
    num_epoch = 60                             # num epochs
    l_rate = 0.001								#learning rate
    lr_step = 15	                       #decay of l_rate after these steps
    w_decay = 0.001                         #Adam
    betas = (0.92, 0.999)                    #Adam
    momentum = 0.9                          # SGD momentum
    weightz = [1558/9247, 7689/9247]         #weights for loss function as per sample size
    statistics_file = "meanAndSD.npy"
    
    #data transformations and augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val", "test"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=b_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        for x in ["train", "val", "test"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
    class_names = image_datasets["train"].classes
    
    #choose GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.show()
        # plt.pause(0.001)  # pause a bit so that plots are updated

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders["train"]))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])

    # writer object for Tensorboard
    writer = SummaryWriter(comment=tb_comment)
    
    #function to plot confusion matrix from a confusion matrix
    def plot_confusion_matrix(
        cm, target_names, title="Confusion matrix", cmap=None, normalize=True
    ):

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap("Blues")

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(
                    j,
                    i,
                    "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            else:
                plt.text(
                    j,
                    i,
                    "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel(
            "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(
                accuracy, misclass
            )
        )
        writer.add_figure(
            "Confusion_Matrix", fig, global_step=None, close=True, walltime=None
        )
        plt.show()
    

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        
        #to load pre-trained weights
        if m_Wts_preload is not None:
            print("Loading locally pre-trained weights...")
            model.load_state_dict(torch.load(m_Wts_preload))

        bestPre_model_wts = copy.deepcopy(model.state_dict())
        bestRec_model_wts = copy.deepcopy(model.state_dict())

        best_precision = 0.0
        best_recall = 0.0

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            # Each epoch has a training and validation phase
            for phase in ["train", "val", "test"]:
                epoch_preds = None
                epoch_labels = None
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        # print(outputs.shape)s
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    if epoch_labels is None:
                        epoch_labels = labels.cpu().numpy()
                    else:
                        epoch_labels = np.concatenate(
                            (epoch_labels, labels.cpu().numpy())
                        )

                    # epoch_labels.append(label)
                    if epoch_preds is None:
                        epoch_preds = preds.cpu().numpy()
                    else:
                        epoch_preds = np.concatenate((epoch_preds, preds.cpu().numpy()))
                    # epoch_preds.append(preds)

                if phase == "train":
                    scheduler.step()

                # calculation of evaluation metrics
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epoch_prec = precision_score(epoch_labels, epoch_preds, average="weighted")
                epoch_rec = recall_score(epoch_labels, epoch_preds, average="weighted")
                epoch_f1 = f1_score(epoch_labels, epoch_preds, average="weighted")
                
                #plot confusion matrix for validation and test sets only 
                if phase in ["val", "test"] and epoch == num_epochs - 1:
                    cm = confusion_matrix(
                        epoch_labels, epoch_preds, labels=None, normalize=None
                    )
                    
                    plot_confusion_matrix(
                        cm,
                        target_names=["notPotH", "potH"],
                        title="Confusion Matrix " + phase + " set",
                        cmap=None,
                        normalize=False,
                    )
                    plot_confusion_matrix(
                        cm,
                        target_names=["notPotH", "potH"],
                        title="Normalized Confusion Matrix " + phase + " set",
                        cmap=None,
                        normalize=True,
                    )

                #write to tensorboard
                writer.add_scalars("loss", {phase: epoch_loss}, epoch)
                writer.add_scalars("precision", {phase: epoch_prec}, epoch)
                writer.add_scalars("recall", {phase: epoch_rec}, epoch)
                writer.add_scalars("f1_score", {phase: epoch_f1}, epoch)
                

                writer.flush()
                print(
                    "{} Loss: {:.4f} Acc: {:.4f} Ep_Prec: {:.4f} Ep_Rec: {:.4f} Ep_F1Score: {:.4f}".format(
                        phase, epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1
                    )
                )

                # deep copy the model
                if phase == "val" and epoch_prec > best_precision:
                    best_precision = epoch_prec
                    bestPre_model_wts = copy.deepcopy(model.state_dict())
                if phase == "val" and epoch_rec > best_recall:
                    best_recall = epoch_rec
                    bestRec_model_wts = copy.deepcopy(model.state_dict())
            if (
                epoch % 3 == 0 or epoch == num_epoch
            ):  # save model weights every 3rd epoch
                torch.save(model.state_dict(), "recentWtz")

            print("Time Elapsed:", (time.time() - since) / 60, " mins")
            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Prec: {:4f} Rec: {:4f}".format(best_precision, best_recall))

        torch.save(bestPre_model_wts, "recentPreWeights")
        torch.save(bestRec_model_wts, "recentRecWeights")
        return model

    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders["val"]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis("off")
                    ax.set_title("predicted: {}".format(class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    model_ft = models.googlenet(pretrained=True)
    #model_ft = models.resnet18(pretrained=True)
    
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.aux_logits = False
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)
    # print(torch.from_numpy(np.float32(weightz)))
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.float32(weightz)).to(device)
    )

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=l_rate, momentum=momentum)
    optimizer_ft = optim.Adam(
        model_ft.parameters(), lr=l_rate, betas=betas, weight_decay=w_decay
    )

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=lr_step, gamma=0.1)

    model_ft = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epoch
    )

    torch.save(model_ft.state_dict(), "recentWtzz")

    visualize_model(model_ft)
    writer.close()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
