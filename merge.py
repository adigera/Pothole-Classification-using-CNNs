# This function is created to merge all the positive and negative images from different
# folders that were downloaded into one positive and negative folders for each
# train and test
# Dataset: https://drive.google.com/drive/u/1/folders/1vUmCvdW3-2lMrhsMbXdMWeLcEz__Ocuy

import numpy as np
from PIL import Image
from glob import iglob
import matplotlib.pyplot as plt
import os

try:
    os.makedirs("train_Positive")
    os.makedirs("train_Negative")
    os.makedirs("test_Positive")
    os.makedirs("test_Negative")
except OSError:
    print("Creation of the directory %s failed")
else:
    print("Successfully created the directory %s ")


def main():
    image_files = list(iglob("**/*.JPG", recursive=True))
    # making folders to copy to
    i = 1
    j = 1
    k = 1
    l = 1
    for im in image_files:
        temp = Image.open(im)
        if ("train" or "tra" or "rain") in im.lower() and (
            "positive" or "posi" or "siti"
        ) in im.lower():
            temp.save("train_Positive/positive" + str(i) + ".JPG")
            i = i + 1
        if ("train" or "tra" or "rain") in im.lower() and (
            "negative" or "gati" or "nega"
        ) in im.lower():
            temp.save("train_Negative/negative" + str(j) + ".JPG")
            j = j + 1
        if (
            "test" or "tes" or "est"
        ) in im.lower():  # and ('positive' or 'posi' or 'siti') in im.lower():
            temp.save("test_Positive/positive" + str(k) + ".JPG")
            k = k + 1
            if k % 100 == 0:
                print("Images Processes:", k)

        if ("test" or "tes" or "est") in im.lower() and (
            "negative" or "gati" or "nega"
        ) in im.lower():
            temp.save("test_Negative/negative" + str(l) + ".JPG")
            l = l + 1


main()
# if __name__=='__main__':
# 	main()
