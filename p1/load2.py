# import cv2
import os


def get_filepaths(path):
    file_paths = []
    count = 0
    for root, directories, files in os.walk(path):
        if count == 5:
            break
        count+=1
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


imlist = get_filepaths("c:/Users/ran/Downloads/sample_drive/cam1")

import os, numpy, PIL
from PIL import Image

# Access all PNG files in directory
# allfiles=os.listdir(os.getcwd())
# # imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]
#
# # Assuming all images are the same size, get dimensions of first image
# w,h=Image.open(imlist[0]).size
# N=len(imlist)

# Create a numpy array of floats to store the average (assume RGB images)
arr=numpy.zeros((2032,2032,3),numpy.float)

# Build up average pixel intensities, casting each image as an array of floats
count = 0
for im in sorted(imlist):
    count+=1
    print (count)
    imarr=numpy.array(Image.open(im),dtype=numpy.float)
    imarrg=numpy.gradient(imarr)
    # print imarr
    imarrg=numpy.array(imarrg)[1]
    print(imarrg)
    arr=arr+imarrg / (len(imlist) / 20)

# Round values in array and cast as 8-bit integer
arr=numpy.array(numpy.round(arr),dtype=numpy.uint8)

# Generate, save and preview final image
out=Image.fromarray(arr,mode="RGB")
out.save("Average.png")
out.show()
# # print(file_list)
# temp = cv2.imread(file_list[0], cv2.IMREAD_GRAYSCALE)
# i = 0
#
# for f in file_list:
#     img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#     temp = temp + img
#     newx, newy = int(img.shape[1] / 4), int(img.shape[0] / 4)
#     newimage = cv2.resize(img, (newx, newy))
#     newtemp = cv2.resize(temp, (newx, newy))
#     cv2.imshow('image', newimage)
#     if i % 100 == 0:
#         cv2.imshow('temp', newtemp)
#         temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
#     cv2.waitKey(1)
#     i += 1
