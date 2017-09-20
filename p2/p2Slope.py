import csv
import numpy as np
import math
import pickle
import os.path

# set to True to test in first 500 data points.
TESTING = False

dirPath = 'data/'
linkPath = 'Partition6467LinkData.csv'
probePath = 'Partition6467ProbePoints.csv'
matchedPath = 'Partition6467MatchedPoints.csv'

linkFile = open(dirPath+linkPath)
linkSlopes = {}
linkIds = []
linkReader = csv.reader(linkFile)
count=0
for row in linkReader:
    #Reading slopes
    linkSlopes[row[0]]=[]
    slopes = []
    altitude = row[14].split('|')[0].split('/')[2]
    if altitude:
        linkSlopes[row[0]].append(float(altitude))
    else:
        linkSlopes[row[0]].append(None)

    for slope in row[16].split('|'):
        if slope:
            dist, degree = slope.split('/')
            slopes.append((float(dist), float(degree)))
    linkSlopes[row[0]].append(slopes)
    linkIds.append(row[0])

    count += 1
    if TESTING and count==10000:
        break
    if count%10000==0:
        print(count)


matchedPointFile = open(dirPath+matchedPath)
matchedPoints = []
matchedPointReader = csv.reader(matchedPointFile)
count=0
for row in matchedPointReader:
    matchedPoints.append((row[8], float(row[5]), float(row[10]), float(row[11])))
    count += 1
    if TESTING and count==10000:
        break
    if count%10000==0:
        print(count)


if os.path.isfile('data/link2Slope.pkl') :
    link2Slope_file = open('data/link2Slope.pkl', 'rb')
    link2Slope = pickle.load(link2Slope_file)
else :
    link2Slope = {}
    for matchedPoint in matchedPoints:
        linkPVID = matchedPoint[0]
        linksDistance = matchedPoint[3]
        if linksDistance > 10:
        	# if point is 10 meters away from link ignore it.
            continue
        linkSlope = linkSlopes[linkPVID]



        refAltitude = linkSlope[0]
        pointAltitude = matchedPoint[1]

        distance = matchedPoint[2]
        #print("distance: "+str(distance))


        if refAltitude is None or distance == 0:
            degree = None
        else:
            diffAltitude = pointAltitude - refAltitude
            # calc slope infomation
            degree = math.degrees(math.atan(diffAltitude/distance))

            #print("degree: "+str(degree))
            if np.absolute(degree)>8:
                degree = None
        
        if distance != 0 and degree:
            link2Slope.setdefault(linkPVID, []).append((distance, degree))

        count += 1
        # debug mode
        if TESTING and count==10000:
            break
        if count%10000==0:
            print(count)

    link2Slope_file = open('data/link2Slope.pkl', 'wb')
    pickle.dump(link2Slope, link2Slope_file)

print("link2Slope")
count=0
for row in link2Slope.items():
    print(row)
    print("\n")
    count+=1
    if count == 10:
        break;

# print(link2Slope.items())



if not os.path.isfile('data/Partition6467LinkSlopedData.csv') :
    with open("data/Partition6467LinkData.csv", "r") as pp, open("data/Partition6467LinkSlopedData.csv", "w") as out:
        for linkPVID,row in zip(linkIds,pp):
            line = row.strip()
            if linkPVID in link2Slope:
                linkSlope = link2Slope[linkPVID]
                linkSlope = sorted(linkSlope, key=lambda x:x[0])
                line += "," + '|'.join(['/'.join(map(str,slope)) for slope in linkSlope])

            out.write(line + "\n")
            if TESTING and idx > 10:
                break


goodcount = 0
count=0
with open("data/Partition6467LinkSlopedData.csv", "r") as pp:
    ppreader = csv.reader(pp)
    for row in ppreader:
        if len(row)>17 and row[16] and row[17]:
            probeSlopes = [tuple(map(float, x.split('/'))) for x in row[17].split('|')]
            linkSlopes = [tuple(map(float, x.split('/'))) for x in row[16].split('|')]
            availSlopes=[]
            linkSlopes = sorted(linkSlopes, key=lambda x:x[0])
            for slope in probeSlopes:
                dist = slope[0]
                degree = slope[1]
                for i in range(0, len(linkSlopes)):
                    if linkSlopes[i][0] >= dist:
                        break
                prevslope = 0
                nextslope = 0
                if i - 1 >= 0 :
                    prevslope = linkSlopes[i - 1][1]
                if i >= 0 :
                    nextslope = linkSlopes[i][1]
                linkslope = (nextslope + prevslope) / 2
                diff = np.absolute(degree - linkslope)
                availSlopes.append((dist, degree, diff, linkslope))

            for slope in availSlopes:
                count+=1
                diff = slope[2]
                if diff < 2:
                    goodcount+=1

print ("%d slopes of %d slopes , %f%%, has less than 2 degrees error" % (goodcount, count, goodcount*100/float(count)))