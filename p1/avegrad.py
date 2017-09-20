"""
Thresholding and mark smears.
Team member: Ran Su, Lingfan Kong, Xiangyu Ma, Jian Zhu

"""
import cv2
import numpy as np
from PIL import Image

# Open Files
ave = cv2.imread("result/c5/Averagecam.png")
ave = np.float32(ave)
grad = cv2.imread('result/c5/AverageGradientcam.png')
grad = np.float32(grad)

# Subtract
out = ave - (grad * 1.5)
print(out)
out = np.array(np.round(out),dtype=np.uint8)
outm = Image.fromarray(out)
outm.show()

# Filter noise
for i in range(10):
	gau = cv2.medianBlur(out,5)

gauo = Image.fromarray(gau)
# gauo = cv2.imread(gauo, )
image = gauo.convert('L')

# Thresholding
gau = np.array(np.where(image < np.mean(image)/2, 255, 0))
gau = np.array(np.round(gau),dtype=np.uint8)

# Filter noise
for i in range(10):
	gau = cv2.medianBlur(gau,9)
	
gaut = Image.fromarray(gau)
gaut.save("result/c5/Thresholded.png")

# print(threshed)
# ret,thresh = cv2.threshold(gau,127,255,0)
im2, contours, hierarchy = cv2.findContours(gau,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(im2)
gauo = Image.fromarray(im2)
gauo.show()
gau = cv2.drawContours(gau, contours, -1, (127), 3)
print(cv2.drawContours.__doc__)

# cnt = np.array(contours)
# cv2.drawContours(gau, cnt, 0, (0,255,0), 3)

print(contours)
gauo = Image.fromarray(gau)
gauo.show()
gauo.save("result/c5/Markedout.png")