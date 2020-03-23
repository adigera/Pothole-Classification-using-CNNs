## Program to find the mean and standard deviation values for the data set

import numpy as np
from glob import iglob
from PIL import Image
import os
import time


def mainNormalize():
    numImages = 0
    numPixPerChannel = 0
    sumR = 0.00
    sumG = 0.00
    sumB = 0.00
    sqSumR = 0.00
    sqSumG = 0.00
    sqSumB = 0.00
    image_files = list(iglob("**/*.JPG", recursive=True))
    start = time.time()

    ############### FINDING MEAN OF EVERY CHANNEL
    for imPath in image_files:
        img = Image.open(imPath)
        imArray = np.divide(np.array(img), 255)
        numPixPerChannel = imArray.shape[0] * imArray.shape[1]
        sumR = sumR + np.sum(imArray[:, :, 0], axis=(0, 1), dtype="float64")
        sumG = sumG + np.sum(imArray[:, :, 1], axis=(0, 1), dtype="float64")
        sumB = sumB + np.sum(imArray[:, :, 2], axis=(0, 1), dtype="float64")
        sqSumR = sqSumR + np.sum(np.square(imArray[:, :, 0]))
        sqSumG = sqSumG + np.sum(np.square(imArray[:, :, 1]))
        sqSumB = sqSumB + np.sum(np.square(imArray[:, :, 2]))
        if numImages % 50 == 0:
            print(
                "Images processed:",
                numImages,
                "   Time:",
                (time.time() - start) / 60,
                " mins",
            )
            print(" ")
        numImages = numImages + 1
    print(
        "TOTAL Images processed:",
        numImages,
        "TOTAL TIME:",
        (time.time() - start) / 60,
        " mins",
    )
    meanR = (sumR / numImages) / numPixPerChannel
    meanG = (sumG / numImages) / numPixPerChannel
    meanB = (sumB / numImages) / numPixPerChannel

    ############ variance = E[x^2] - E[x]^2
    sdR = np.sqrt(((sqSumR / numImages) / numPixPerChannel) - np.square(meanR))
    sdG = np.sqrt(((sqSumG / numImages) / numPixPerChannel) - np.square(meanG))
    sdB = np.sqrt(((sqSumB / numImages) / numPixPerChannel) - np.square(meanB))

    MeanSD = [[meanR, meanG, meanB], [sdR, sdG, sdB]]
    print(" ")
    print(MeanSD)
    np.save("meanAndSD", MeanSD)
    print(" ")


mainNormalize()
