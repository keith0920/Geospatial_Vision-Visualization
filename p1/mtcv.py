from threading import Thread
import os, numpy, PIL
from PIL import Image
import time
import logging
import cv2
import numpy as np

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

def cart(id):
    count = 0
    for im in simlist[int(id*(listlen / NUM_OF_THREADS)) : int((id+1)*(listlen/ NUM_OF_THREADS))]:
        count+=1
        # print (count)
        logger.debug('longging %s', str(id) + '#' + str(count))
        # imarr=numpy.array(Image.open(im).convert('L'),dtype=numpy.float)
        # image = Image.open(im).convert('L')
        # image.show()
        # imarrg=numpy.gradient(imarr)
        # print(len(imarrg))
        # imarrg=numpy.array(imarrg)
        # print(numpy.array(imarrg)[1])
        img = cv2.imread(im)
        img = np.float32(img)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # print(magarr[id].shape)
        # print(((np.array(mag) * 255) / listlen).shape)
        magarr[id] = magarr[id] + ((np.abs(np.array(mag))) / listlen)
        
        # rgx = numpy.array(numpy.round(mag * 255),dtype=numpy.uint8)
        # out=Image.fromarray(rgx)
        # out.show()
        # grax[id] = grax[id] + numpy.array(imarrg)[0] / listlen
        # gray[id] = gray[id] + numpy.array(imarrg)[1] / listlen
        # arr[id] = arr[id] + imarr / listlen
    

if __name__ == '__main__':
    NUM_OF_THREADS = 3
    imlist = get_filepaths("C:/Users/ran/Downloads/sample_drive/cam")
    
    simlist = sorted(imlist)[:100]
    listlen = len(simlist)
    IMAGE_H, IMAGE_W = Image.open(simlist[0]).size
    COLOR_D = tuple((IMAGE_H, IMAGE_W))
    # COLOR_D = tuple((IMAGE_H, IMAGE_W, 3))
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    arr=[numpy.zeros(COLOR_D,numpy.float)]* NUM_OF_THREADS
    # gradient
    magarr = [numpy.zeros((IMAGE_H, IMAGE_W, 3),numpy.float)]* NUM_OF_THREADS
    
    grax = [numpy.zeros(COLOR_D,numpy.float)]* NUM_OF_THREADS
    gray = [numpy.zeros(COLOR_D,numpy.float)]* NUM_OF_THREADS
    
    start = time.time()
    carts = [0] * NUM_OF_THREADS
    for i in range(0,  NUM_OF_THREADS):
        carts[i] = Thread(target=cart, args=[i])
        carts[i].start()
    
    # allarr = numpy.zeros(COLOR_D,numpy.float)
    # allx = numpy.zeros(COLOR_D,numpy.float)
    # ally = numpy.zeros(COLOR_D,numpy.float)
    allmag = numpy.zeros((IMAGE_H, IMAGE_W, 3),numpy.float)
    for i in range(0,  NUM_OF_THREADS):
        carts[i].join()


    for i in range(0,  NUM_OF_THREADS):
        # allarr = allarr + arr[i]
        # allx = allx + grax[i]
        # ally = ally + gray[i]
        allmag = allmag + magarr[i]

    # print(ally)

    # arryyy=numpy.array(numpy.round(ally),dtype=numpy.uint8)
    # arrxxx=numpy.array(numpy.round(allx),dtype=numpy.uint8)

    # grad = numpy.sqrt(numpy.power(arrxxx, 2) + numpy.power(arryyy, 2))
    # gradccc = numpy.array(numpy.round(grad * (10 * (1 + len(imlist) / 40))),dtype=numpy.uint8)
    # allarr = numpy.array(numpy.round(allarr),dtype=numpy.uint8)
    print(time.time()-start)
    # out=Image.fromarray(arryyy, mode="L")
    # outg = Image.fromarray(arrxxx, mode="L")    
    # outc = Image.fromarray(gradccc, mode="L")
    # outg.save("allx.png")
    # outc.save("outc.png")
    # out.save("ally.png")
    # out.show()
    # outg.show()
    # outc.show()

    # mean = np.mean(allmag)
    # median = float(np.median(allmag))*2
    # print(mean)
    # print(median)
    # worked = np.where(allmag < mean*3, allmag, 255)
    # print(worked)

    arrmag =  numpy.array(numpy.round(allmag),dtype=numpy.uint8)
    outm = Image.fromarray(arrmag)
    outm.show()
    # gau = cv2.medianBlur(arrmag,3)
    # gauo = Image.fromarray(gau)
    # gauo.show()

