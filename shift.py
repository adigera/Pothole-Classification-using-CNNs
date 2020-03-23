## Program to shift randomly selected images from original dataset to form validation set or test set


import random, os
import shutil

# no. of images to be shifted
numPosImages = 200
numNegImages = 800

# source path to move from
posDirSrc = r"actual_data/dev/potH/"
negDirSrc = r"actual_data/dev/notPotH/"

# destination path to move to
posDirDes = r"actual_data/test/potH/"
negDirDes = r"actual_data/test/notPotH/"

# move positive images
if numPosImages > 0:
    for i in range(0, numPosImages):
        random_filename = random.choice(
            [
                x
                for x in os.listdir(posDirSrc)
                if os.path.isfile(os.path.join(posDirSrc, x))
            ]
        )
        os.rename(posDirSrc + random_filename, posDirDes + random_filename)
print("Positive done")

# move negative images
if numNegImages > 0:
    for i in range(0, numNegImages):
        if i % 100 == 0:
            print("Images done:", i)
        random_filename = random.choice(
            [
                x
                for x in os.listdir(negDirSrc)
                if os.path.isfile(os.path.join(negDirSrc, x))
            ]
        )
        os.rename(negDirSrc + random_filename, negDirDes + random_filename)
