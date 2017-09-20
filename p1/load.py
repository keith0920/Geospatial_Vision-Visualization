import cv2
import os


def get_filepaths(path):
    file_paths = []

    for root, directories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath) 

    return file_paths

file_list = get_filepaths("c:/Users/ran/Downloads/sample_drive/cam_3")

# print(file_list)
temp = cv2.imread(file_list[0], cv2.IMREAD_GRAYSCALE)
i = 0

for f in file_list:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    temp = temp + img
    newx, newy = int(img.shape[1]/4), int(img.shape[0]/4)
    newimage = cv2.resize(img, (newx, newy))
    newtemp = cv2.resize(temp, (newx, newy))
    cv2.imshow('image', newimage)
    if i % 100 == 0:
        cv2.imshow('temp', newtemp)
        temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    cv2.waitKey(1)
    i += 1
    