'''
Get image

'''

import math
import urllib.request
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import sys


EARTHRADIUS = 6378137
MINLATITUDE = -85.05112878
MAXLATITUDE = 85.05112878
MINLONGITUDE = -180
MAXLONGITUDE = 180

MAXPIXELS = 2560

BASEURL = "http://ecn.t0.tiles.virtualearth.net/tiles/h{}.png?g=392"

#from https://msdn.microsoft.com/en-us/library/bb259689.aspx
def clip(n, minValue, maxValue):
    return min(max(n, minValue), maxValue)

def mapSize(levelOfDetail):
    return 256 << levelOfDetail

def groundResolution(latitude, levelOfDetail):
    latitude = clip(latitude, MINLATITUDE, MAXLATITUDE)
    return math.cos(latitude * math.pi/ 180) * 2 * math.pi * EARTHRADIUS / mapSize(levelOfDetail)

def mapScale(latitude, levelOfDetail, screenDpi):
    return groundResolution(latitude, levelOfDetail) * screenDpi / 0.0254

def latLongToPixelXY(latitude, longitude, levelOfDetail):
    latitude = clip(latitude, MINLATITUDE, MAXLATITUDE)
    longitude = clip(longitude, MINLONGITUDE, MAXLONGITUDE)

    x = (longitude + 180) / 360
    sinLatitude = math.sin(latitude * math.pi / 180)
    y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

    map_Size = mapSize(levelOfDetail)

    return int(clip(x * map_Size + 0.5, 0, map_Size - 1)), int(clip(y * map_Size + 0.5, 0, map_Size - 1))

# def pixelXYToLatLong(pixelX, pixelY, levelOfDetail):
#     map_Size = mapSize(levelOfDetail)
#     x = (clip(pixelX, 0, map_Size - 1) / map_Size) - 0.5
#     y = 0.5 - (clip(pixelY, 0, map_Size - 1) / map_Size)

#     return (90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi), 360 * x

def pixelXYToTileXY(pixelX, pixelY):
    return int(pixelX / 256), int(pixelY / 256)

# def tileXYToPixelXY(tileX, tileY):
#     return tileX * 256, tileY * 256

def tileXYToQuadKey(tileX, tileY, levelOfDetail):
    quadKey = ""
    for i in range(levelOfDetail, 0, -1):
        digit = "0"
        mask = 1 << (i - 1)
        if (tileX & mask) != 0:
            digit = int(digit) + 1
        if (tileY & mask) != 0:
            digit = int(digit) + 2
        quadKey = quadKey + str(digit)
    return quadKey

# def quadKeyToTileXY(quadKey):
#     tileX = tileY = 0
#     levelOfDetail = len(quadKey)
#     for i in range(levelOfDetail, 0, -1):
#         mask = 1 << (i - 1)
#         if quadKey[levelOfDetail - i] == "0":
#             break
#         elif quadKey[levelOfDetail - i] == "1":
#             tileX = tileX | mask
#             break
#         elif quadKey[levelOfDetail - i] == "2":
#             tileY = tileY | mask
#             break
#         elif quadKey[levelOfDetail - i] == "3":
#             tileX = tileX | mask
#             tileY = tileY | mask
#             break
#     return tileX, tileY, levelOfDetail

'''
By Ran Su

'''
def latLongToQuadKey(latitude, longitude, levelOfDetail):
    '''
    wrap all above functions up
    '''
    pixelX, pixelY = latLongToPixelXY(latitude, longitude, levelOfDetail)
    # print("pixelX {}, pixelY {}".format(pixelX, pixelY))
    tileX, tileY = pixelXYToTileXY(pixelX, pixelY)
    # print("tileX {}, tileY {}".format(tileX, tileY))

    # print("x {}, y {}".format(pixelX - tileX * 256, pixelY - tileY * 256))
    return tileXYToQuadKey(tileX, tileY, levelOfDetail)

def testLevelOfDetial(lat1, lon1, lat2, lon2):
    '''
    find the best level of detial, the output image will not exceed MAXPIXELS(default 10240)xMAXPIXELS pixels(default 40x40 tiles)
    '''
    largest = max(abs(lat1 - lat2), abs(lon1 - lon2))
    # print(largest)
    i = 23
    while (urllib.request.urlopen(BASEURL.format(latLongToQuadKey(lat1, lon1, i))).getheader("Content-Length") == "1033") or (largest * 0.75 * 2**i > MAXPIXELS):
        # print(urllib.request.urlopen(BASEURL.format(latLongToQuadKey(latitude1, longitude1, i))).getheader("Content-Length") == "1033")
        # print(largest * 5.686 * 2**i)
        i -= 1
    while (urllib.request.urlopen(BASEURL.format(latLongToQuadKey(lat1, lon2, i))).getheader("Content-Length") == "1033") or (largest * 0.75 * 2**i > MAXPIXELS):
        # print(urllib.request.urlopen(BASEURL.format(latLongToQuadKey(latitude1, longitude1, i))).getheader("Content-Length") == "1033")
        # print(largest * 5.686 * 2**i)
        i -= 1
    while (urllib.request.urlopen(BASEURL.format(latLongToQuadKey(lat2, lon1, i))).getheader("Content-Length") == "1033") or (largest * 0.75 * 2**i > MAXPIXELS):
        # print(urllib.request.urlopen(BASEURL.format(latLongToQuadKey(latitude1, longitude1, i))).getheader("Content-Length") == "1033")
        # print(largest * 5.686 * 2**i)
        i -= 1
    while (urllib.request.urlopen(BASEURL.format(latLongToQuadKey(lat2, lon2, i))).getheader("Content-Length") == "1033") or (largest * 0.75 * 2**i > MAXPIXELS):
        # print(urllib.request.urlopen(BASEURL.format(latLongToQuadKey(latitude1, longitude1, i))).getheader("Content-Length") == "1033")
        # print(largest * 5.686 * 2**i)
        i -= 1
    return i

def fetch(url):
    '''
    return a image as np.array
    '''
    response = urllib.request.urlopen(url)
    return np.asarray(Image.open(BytesIO(response.read())))

def nextH(string):
    output = ""
    last = string[-1]
    # print(last)
    try:
        if last == "0" or last == "2":
            output = str(int(string) + 1)
            if string[0] == "0":
                output = "0" + output
        if last == "1":
            output = nextH(string[:-1]) + "0"
        if last == "3":
            output = nextH(string[:-1]) + "2"
    except:
        raise
    return output

# print("0302222201")
# print(nextH("0302222201"))

def nextV(string):
    level = len(string)
    output = []
    temp = 0
    for i in range(level):
        level_i = int(string[level-1-i])
        if i == 0:
            level_i += 2
        else:
            level_i += temp
        temp = int((level_i/4))*2
        output.append(str(level_i%4))
    output.reverse()
    output = ''.join(output)
    return output

def getMatrix(lat1, lon1, lat2, lon2, levelOfDetail):
    lat1, lat2 = max(lat1, lat2), min(lat1, lat2)
    if lon1 < 0 and lon2 < 0:
        lon1, lon2 = min(lon1, lon2), max(lon1, lon2)
    else:
        lon1, lon2 = max(lon1, lon2), min(lon1, lon2)
    limit = int(MAXPIXELS / 256) + 1
    rt = latLongToQuadKey(lat1, lon2, levelOfDetail)
    print("rt = " + rt)
    lb = latLongToQuadKey(lat2, lon2, levelOfDetail)
    v = h = limit
    k = np.empty((limit + 1, limit + 1), dtype=np.ndarray)
    k[0][0] = str(latLongToQuadKey(lat1, lon1, levelOfDetail))
    im = np.empty((limit + 1, limit + 1), dtype=np.ndarray)
    im[0][0] = fetch(BASEURL.format(str(k[0][0])))
    col = np.empty(limit, dtype=np.ndarray)

    for i in range(0, limit):
        if i > v:
            break
        col[i] = im[i][0]
        for j in range(1, limit):
            if j > h:
                break
            print("fetching x:{}th y:{}th image".format(i, j))
            k[i][j] = nextH(k[i][j - 1])
            print("k[i][j] {} k[i][j - 1] {}".format(k[i][j], k[i][j - 1]))
            im[i][j] = fetch(BASEURL.format(k[i][j]))

            # horizontally stack
            col[i] = np.column_stack((col[i], im[i][j]))
            if k[i][j] == rt:
                h = j
            if k[i][j] == lb:
                v = i
                print("v = " + str(v))
        k[i + 1][0] = nextV(k[i][0])
        im[i + 1][0] = fetch(BASEURL.format(k[i + 1][0]))

    #vertically stack
    img = col[0]
    for i in range(1, len(col)):
        # print(type(col[i]))
        if (type(col[i]) is type(col[0])):
            # print(col[i].shape)
            img = np.row_stack((img, col[i]))

    # prepare final cut
    pixelX, pixelY = latLongToPixelXY(lat1, lon1, levelOfDetail)
    tileX, tileY = pixelXYToTileXY(pixelX, pixelY)
    x1, y1 = pixelX - tileX * 256, pixelY - tileY * 256
    # print("x {}, y {}".format(x1, y1))
    pixelX, pixelY = latLongToPixelXY(lat2, lon2, levelOfDetail)
    tileX, tileY = pixelXYToTileXY(pixelX, pixelY)
    x2, y2 = pixelX - tileX * 256, pixelY - tileY * 256

    #cut
    return img[y1:-(256 - y2), x1:-(256 - x2)]



if __name__ == "__main__":
    latitude1 = 41.8362778
    longitude1 = -87.6296155
    latitude2 = 41.897076
    longitude2 = -87.601553
    outfile = 'test'
    argv = sys.argv
    if len(argv) != 6 :
        print('Usage: python3.x getqkey.py [lat1] [long1] [lat2] [long2] [output] ')
        print('Using default value')
    else :
        latitude1 = float(argv[1])
        longitude1 = float(argv[2])
        latitude2 = float(argv[3])
        longitude2 = float(argv[4])
        outfile = argv[5]

    lvl = testLevelOfDetial(latitude1, longitude1, latitude2, longitude2)

    data = getMatrix(latitude1, longitude1, latitude2, longitude2, lvl)
    print(data.shape)
    plt.imshow(data)
    plt.show()
    scipy.misc.imsave(str(outfile)+'.jpg', data)
