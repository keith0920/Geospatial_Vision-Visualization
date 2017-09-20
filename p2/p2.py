import numpy as np
import csv
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from operator import itemgetter
import pickle
import os.path

# set to True to test in first 500 data points.
TESTING = False

def calculateDistance(base, sideLeft, sideRight):
    sideLeft2 = sideLeft ** 2
    sideRight2 = sideRight ** 2
    base2 = base ** 2
    ret = np.zeros_like(base, dtype=np.float)
    mask = np.ones_like(base, dtype=np.bool)
    temp = np.logical_and(base == sideLeft+sideRight, mask)
    mask = np.logical_xor(mask, temp)
    ret[temp] = 0
    temp = np.logical_and(base < 1e-10, mask)
    mask = np.logical_xor(mask, temp)
    ret[temp] = sideLeft[temp]
    temp = np.logical_and(sideLeft2 > base2 + sideRight2, mask)
    mask = np.logical_xor(mask, temp)
    ret[temp] = sideRight[temp]
    temp = np.logical_and(sideRight2 > base2 + sideLeft2, mask)
    mask = np.logical_xor(mask, temp)
    ret[temp] = sideLeft[temp]

    perimeter = (base[mask] + sideLeft[mask] + sideRight[mask]) * 0.5
    area = np.sqrt(perimeter * (perimeter - sideLeft[mask]) * (perimeter - sideRight[mask]) * (perimeter - base[mask]))
    ret[mask] = 2 * area / base[mask]
 
    return ret

# From StackOverFlow
def haversine_np(probe, link_points):
    delta = probe - link_points
    d_lat = delta[:, 0]
    d_lon = delta[:, 1]
    # if type(probe) != np.ndarray:
    #     probe = np.array(probe)

    if len(probe.shape) == 1:
        lat1 = probe[0]
    else:
        lat1 = probe[:, 0]
    lat2 = link_points[:, 0]
    temp = np.sin(d_lat/2.0)**2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(d_lon/2.0)**2
    ratio = 2 * np.arcsin(np.sqrt(temp))

    return 6371004 * ratio


dirPath = 'data/'
linkPath = 'Partition6467LinkData.csv'
probePath = 'Partition6467ProbePoints.csv'


if os.path.isfile('data/probePoints.pkl') and os.path.isfile('data/probeIds.pkl'):
    probePoints_file = open('data/probePoints.pkl', 'rb')
    probePoints = pickle.load(probePoints_file)
    probeIds_file = open('data/probeIds.pkl', 'rb')
    probeIds = pickle.load(probeIds_file)
else:
    probeFile = open(dirPath+probePath)
    probePoints = []
    probeIds = []
    probeTimes = []
    probeReader = csv.reader(probeFile)
    count=0
    for row in probeReader:
        #Reading latitude, longitude
        point = (float(row[3]), float(row[4]))
        probePoints.append(point)

        #Reading probe id
        probeIds.append(row[0])

        #Reading probe time
        timeFormat = "%m/%d/%Y %I:%M:%S %p"
        probeTimes.append(datetime.strptime(row[1], timeFormat))
        count += 1
        if TESTING and count == 500:
            break
        if count%10000==0:
            print(count)
    probePoints_file = open('data/probePoints.pkl', 'wb')
    pickle.dump(probePoints, probePoints_file)
    probeIds_file = open('data/probeIds.pkl', 'wb')
    pickle.dump(probeIds, probeIds_file)

print("ProbePoints: ")
print(probePoints[:5])
oriprobePoints = probePoints


if os.path.isfile('data/linkShapes.pkl') and os.path.isfile('data/linkIds.pkl') and os.path.isfile('data/linkLengths.pkl'):
    linkShapes_file = open('data/linkShapes.pkl', 'rb')
    linkShapes = pickle.load(linkShapes_file)
    linkIds_file = open('data/linkIds.pkl', 'rb')
    linkIds = pickle.load(linkIds_file)
    linkLengths_file = open('data/linkLengths.pkl', 'rb')
    linkLengths = pickle.load(linkLengths_file)
else:
    linkFile = open(dirPath+linkPath)
    linkShapes = []
    linkIds = []
    linkLengths = []
    linkReader = csv.reader(linkFile)
    count=0
    for row in linkReader:
        #Reading shapes
        shapes = []
        for point in row[14].split('|'):
            latitude, longitude, _ = point.split('/')
            shapes.append((float(latitude), float(longitude)))
        linkShapes.append(shapes)

        #Reading link id
        linkIds.append(row[0])

        #Reading link length
        linkLengths.append(float(row[3]))
        count+=1
        if TESTING and count==500:
            break
        if count%10000==0:
            print(count)
    linkShapes_file = open('data/linkShapes.pkl', 'wb')
    pickle.dump(linkShapes, linkShapes_file)
    linkIds_file = open('data/linkIds.pkl', 'wb')
    pickle.dump(linkIds, linkIds_file)
    linkLengths_file = open('data/linkLengths.pkl', 'wb')
    pickle.dump(linkLengths, linkLengths_file)



print("LinkShapes: ")
print(linkShapes[:5])


if os.path.isfile('data/linkPoints.pkl') and os.path.isfile('data/linkMaps.pkl'):
    linkPoints_file = open('data/linkPoints.pkl', 'rb')
    linkPoints = pickle.load(linkPoints_file)
    linkMaps_file = open('data/linkMaps.pkl', 'rb')
    linkMaps = pickle.load(linkMaps_file)
else:
    #Finding links' index, and storing them in a dictionary
    linkMaps = dict()
    count=0
    for index, link in enumerate(linkShapes):
        for linkPoint in link:
            linkMaps[linkPoint] = linkMaps.get(linkPoint, [])
            linkMaps[linkPoint].append(index)
        count+=1
        if TESTING and count==500:
            break
        if count%10000==0:
            print(count)

    linkPoints = list(linkMaps.keys())
    linkMaps_file = open('data/linkMaps.pkl', 'wb')
    pickle.dump(linkMaps, linkMaps_file)
    linkPoints_file = open('data/linkPoints.pkl', 'wb')
    pickle.dump(linkPoints, linkPoints_file)

    


print("LinkMaps: length")
print(len(linkMaps))

print("linkPoints: ")
print(linkPoints[:5])

if os.path.isfile('data/potentialLinks.pkl'):
    potentialLinks_file = open('data/potentialLinks.pkl', 'rb')
    potentialLinks = pickle.load(potentialLinks_file)
else:
    probePoints = np.array(probePoints)
    linkPoints = np.array(linkPoints)
    probePointsRad = np.deg2rad(probePoints)
    linkPointsRad = np.deg2rad(linkPoints)

    # print("probePointsRad: ")
    # print(probePointsRad)

    # print("linkPointsRad: ")
    # print(linkPointsRad)

    nearestNeighbors = NearestNeighbors(n_neighbors=100, algorithm="kd_tree").fit(linkPointsRad)
    potentialLinks = nearestNeighbors.kneighbors(probePointsRad, return_distance=False)
    potentialLinks = [set([bel for lat, lon in linkPoints[pl]for bel in linkMaps[lat, lon]]) for pl in potentialLinks]

    potentialLinks_file = open('data/potentialLinks.pkl', 'wb')
    pickle.dump(potentialLinks, potentialLinks_file)

print("potentialLinks: ")
print(potentialLinks[:5])


if os.path.isfile('data/probe2Links.pkl'):
    probe2Links_file = open('data/probe2Links.pkl', 'rb')
    probe2Links = pickle.load(probe2Links_file)
else:
    probe2Links = []
    count=0
    for probe, links in zip(oriprobePoints, potentialLinks):
        linkDis = dict()
        count+=1
        if count%10000==0:
            # break
            print(count)

        for linkId in links:
            linkPointRad = np.deg2rad(linkShapes[linkId])
            probeRad = np.deg2rad(probe)
            base = haversine_np(linkPointRad[:-1,:], linkPointRad[1:,:])
            side = haversine_np(probeRad, linkPointRad)
            sideLeft = side[:-1]
            sideRight = side[1:]

            link_dis = calculateDistance(base, sideLeft, sideRight)
            link_dis=np.min(link_dis)
            # record distance < 100 links
            if link_dis < 100:
                linkDis[linkId] = link_dis
        if len(linkDis) == 0:
            probe2Links.append(None)
        else:
            probe2Link = min(linkDis.items(), key=itemgetter(1))
            probe2Links.append(probe2Link)
    
    probe2Links_file = open('data/probe2Links.pkl', 'wb')
    pickle.dump(probe2Links, probe2Links_file)

print("probe2Links: ")
print(probe2Links[:5])
    

if os.path.isfile('data/drivingInfo.pkl'):
    drivingInfo_file = open('data/drivingInfo.pkl', 'rb')
    drivingInfo = pickle.load(drivingInfo_file)
else:
    drivingInfo = []
    for i, (probe, link, pid) in enumerate(zip(oriprobePoints, probe2Links, probeIds)):
        if link is not None:
            linkId, linkDistance = link

            linkPointRad = np.deg2rad(linkShapes[linkId])
            probeRad = np.deg2rad(probe)

            base = haversine_np(linkPointRad[:-1,:], linkPointRad[1:,:])
            side = haversine_np(probeRad, linkPointRad)
            sideLeft = side[:-1]
            sideRight = side[1:]

            linkDistance = calculateDistance(base, sideLeft, sideRight)
            minidx = np.argmin(linkDistance)
            linkV = np.array(linkShapes[linkId][minidx + 1]) - np.array(linkShapes[linkId][minidx])

            cos = -100
            if i > 0 and probeIds[i - 1] == pid:
                porbeV  =  np.array(probe) - np.array(oriprobePoints[i - 1])
                linkV /= np.linalg.norm(linkV)
                porbeV /= np.linalg.norm(porbeV)
                cos = np.dot(linkV, porbeV)

            if i < len(probe) - 1 and probeIds[i + 1] == pid:
                porbeV  =  np.array(oriprobePoints[i + 1]) - np.array(probe)
                linkV /= np.linalg.norm(linkV)
                porbeV /= np.linalg.norm(porbeV)
                cos1 = np.dot(linkV, porbeV)
                if cos == -100 or np.abs(cos1) > np.abs(cos):
                    cos = cos1
            if cos == -100:
                cos = 1
            drivingInfo.append(cos)
        else:
            drivingInfo.append(None)
        if i % 10000 == 0:
            print (i)

    drivingInfo_file = open('data/drivingInfo.pkl', 'wb')
    pickle.dump(drivingInfo, drivingInfo_file)

print("drivingInfo: ")
print(drivingInfo[:5])

if os.path.isfile('data/distances.pkl'):
    distances_file = open('data/distances.pkl', 'rb')
    distances = pickle.load(distances_file)
else:
    count = 0
    distances = []
    for probe, link in zip(oriprobePoints, probe2Links):
        if link is not None:
            linkId, linkDistance = link

            linkPointRad = np.deg2rad(linkShapes[linkId])
            probeRad = np.deg2rad(probe)

            base = haversine_np(linkPointRad[:-1,:], linkPointRad[1:,:])
            side = haversine_np(probeRad, linkPointRad)
            sideLeft = side[:-1]
            sideRight = side[1:]

            linkDistance = calculateDistance(base, sideLeft, sideRight)
            minidx = np.argmin(linkDistance)
            length = (base[minidx] ** 2 + sideLeft[minidx] ** 2  - sideRight[minidx] ** 2) / (2 * base[minidx])
            if length > base[minidx]:
                length = base[minidx]
            if length < 0:
                length = 0
            refDistance = np.sum(base[:minidx]) + length
            refDistance = refDistance / np.sum(base) * linkLengths[linkId]
            linksDistance = linkDistance[minidx]

            distances.append((refDistance, linksDistance))
            count += 1
            if count % 10000 == 0:
                print(count)
        else:
            distances.append(None)

    distances_file = open('data/distances.pkl', 'wb')
    pickle.dump(distances, distances_file)

print("distances: ")
print(distances[:5])


with open("data/Partition6467ProbePoints.csv", "r") as pp, open("data/Partition6467MatchedPoints.csv", "w") as out:
    for idx, (line, nearest, di, dist) in enumerate(zip(pp, probe2Links, drivingInfo, distances)):
        if nearest is not None:
            nearest_idx, nearest_dis = nearest
            line = line.strip()
            dis_ref, dis_nonref = dist
            direction = "F" if di > 0 else "T" 
            link_pvid = linkIds[nearest_idx]
            line += "," + ",".join([link_pvid, direction, str(dis_ref), str(dis_nonref)])
            # print (line)
            out.write(line + "\n")
        if TESTING and idx > 10:
            break
