from __future__ import print_function

import numpy as np
import argparse
import cv2    
from scipy import signal
import os 
import sys
import random
from skimage import measure

CONTOUR_WIDTH = 5

class ImageReader():
	imageList = []
	bVideo = False
	index = 0
	cap = None
	def __init__(self, path):
		self.imageList = []
		self.index = 0
		self.imageList = [ path+'/'+f for f in os.listdir(path) if f.endswith(".jpg") or f.endswith(".png") ]

	def read(self):
		if (self.index >= len(self.imageList)):
			return (False, None, None)
		img = cv2.imread(self.imageList[self.index])
		imageName = self.imageList[self.index]
		self.index = self.index + 1
		return (True, img, imageName)

	def previous(self):
		if self.index > 0 : self.index = self.index - 1

def showResizeImg(img, name, waitMS, x, y, width=1280, height=720):
	(h,w) = img.shape[:2]
	r = min([float(width)/w, float(height)/h])
	(w, h) = (int(r * w), int(r * h))

	img = cv2.resize(img, (w, h))
	cv2.imshow(name, img)
	cv2.moveWindow(name, x, y)
	key = cv2.waitKey(waitMS)

	return key & 0xFF


def blur_img(img, method):
	if (method == 'bilateral'):
		diameter = 9
		sigmaColor = 21
		sigmaSpace = 7
		blur = cv2.bilateralFilter(img, diameter, sigmaColor, sigmaSpace)
	else:
		blur = img.copy()
	return blur

def threshold_img(img, method):
	if (method == 'adaptive'):
		thresholded = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
	elif (method == 'OTSU'):
		T, thresholded = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
	else:
		if (method != None):
			try:
				T = int(method)
			except:
				T = 128
		T, thresholded = cv2.threshold(img, T, 255, cv2.THRESH_BINARY_INV)
	return thresholded

def morphological_process(img, method, kernelSize):
	if (method == 'closing'):
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
		closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		return closing
	elif (method == 'erode'):
		eroded = cv2.erode(img, None, iterations=1)
		return eroded
	elif (method == 'dilate'):
		dilated = cv2.dilate(eroded, None, iterations=1)
		return dilated
	return img.copy()

def find_toRemoveList(subList):

	if len(subList) == 1:
		return list(subList)

	voteList = []
	max_vote = None	
	for ((x, y, w, h), cnt) in subList:
		area = cv2.contourArea(cnt)

		if len(voteList) == 0:
			voteList.append(((x, y, w, h), area, 1))
			max_vote = voteList[0]
		else:
			found_v = False
			for v in voteList:
				((vx, vy, vw, vh), v_area, v_vote) = v
				if (w < vw-20) or (w > vw+20):
					continue
				if (h < vh-20) or (h > vh+20):
					continue
				voteList[voteList.index(v)] = ((vx, vy, vw, vh), v_area, v_vote+1)
				(_, _, vote_cnt) = max_vote
				if (vote_cnt < (v_vote+1)):
					max_vote = ((vx, vy, vw, vh), v_area, v_vote+1)
				found_v = True
				break
			if (found_v == False):
				voteList.append(((x, y, w, h), area, 1))

	toRemove = []
	((vx, vy, vw, vh), v_area, v_vote) = max_vote
	if (v_vote == 1):
		return list(subList)

	#print('------------')
	for ((x, y, w, h), cnt) in subList:
		if (w < vw-20) or (w > vw+20) or \
						(h < vh-20) or (h > vh+20) :
			toRemove.append(((x, y, w, h), cnt))
			#print('removed', (x, y, w, h))
			continue
		#print((x, y, w, h))	

	toRemove = sorted(toRemove, key=lambda b:b[0], reverse=False)
	return toRemove

def filter_contours(img, contours, virtualize=True):
	
	clone = img.copy()
	if (virtualize == True):
		cv2.drawContours(clone, contours, -1, (0, 0, 0), CONTOUR_WIDTH)
		showResizeImg(clone, 'original contour', 0, 0, 0)

	filtered = []
	filteredBox = []
	for c in contours:
		area = cv2.contourArea(c)
		if (area < 200):
		#if (False):
			continue
		else:

			(x, y, w, h) = cv2.boundingRect(c)
			rectArea = w*h
			solidity = area/float(rectArea)
			ar = float(w)/float(h)
			#if (solidity > 0.3) and (ar > 0.4) and (ar < 0.9):
			if (ar > 0.4) and (ar < 0.9):
				color = (0, 255, 0)
				filtered.append(c)
				filteredBox.append((x, y, w, h))
			else:
				color = (0, 0, 255)

	if (virtualize == True):
		cv2.drawContours(clone, filtered, -1, (255, 0, 0), CONTOUR_WIDTH)
		showResizeImg(clone, 'remove unlikely: keep blue', 0, 0, 0)

	(filtered, filteredBox) = zip(*sorted(zip(filtered, filteredBox),
								key=lambda b:b[1][1], reverse=False))
	candidateList = []
	subList = []
	(lastX, lastY, lastW, lastH) = (0, 0, 0, 0)
	for ((x, y, w, h), cnt) in zip(filteredBox, filtered):
		if (y > (lastY+20)) or (y < (lastY-20)):
			if len(subList) > 0:
				subList = sorted(subList, key=lambda a:a[0], reverse=False)
				candidateList.append(list(subList))
			subList = []
		subList.append(((x, y, w, h), cnt))
		(lastX, lastY, lastW, lastH) = (x, y, w, h)

	if (virtualize == True):
		for subList in candidateList:
			#print('**********')
			color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
			for ((x, y, w, h), cnt) in subList:
				#print((x, y, w, h))
				cv2.drawContours(clone, [cnt], -1, color, CONTOUR_WIDTH)
		showResizeImg(clone, 'sorted by y position', 0, 0, 0)

	filtered = []
	#clone = img.copy()	
	for subList in candidateList:
		toRemove = find_toRemoveList(subList)
		subList = [item for item in subList if item not in toRemove]
		if (virtualize == True):
			for ((x, y, w, h), cnt) in subList:
				cv2.drawContours(clone, [cnt], -1, (0, 255, 0), CONTOUR_WIDTH)
			for ((x, y, w, h), cnt) in toRemove:
				cv2.drawContours(clone, [cnt], -1, (0, 0, 255), CONTOUR_WIDTH)
	if (virtualize == True):
		showResizeImg(clone, 'red will be removed', 0, 0, 0)	

	return candidateList

def draw_result(img, candidateList):
	clone = img.copy()
	for subList in candidateList:
		if len(subList) == 1:
			continue
		x1 = 0
		y1 = 0
		y0, x0 = img.shape[:2]
		for ((x, y, w, h), cnt) in subList:
			x0 = min([x, x0])
			y0 = min([y, y0])
			x1 = max([(x+w), x1])
			y1 = max([(y+h), y1])
			ar = float(x1-x0)/(y1-y0)
		if ar > 1.2 and ar < 8:
			for ((x, y, w, h), cnt) in subList:
				cv2.drawContours(clone, [cnt], -1, (0, 255, 0), CONTOUR_WIDTH)	
				cv2.rectangle(clone, (x0, y0), (x1, y1), (255, 0, 0), CONTOUR_WIDTH)
	showResizeImg(clone, 'Result : q to quit, p to repeat', 1, 0, 0)

def extract_blobs(img, visualize = False):
	labels = measure.label(img, neighbors=8, background=0)
	mask = np.zeros(img.shape, dtype="uint8")
	print("[INFO] found {} blobs".format(len(np.unique(labels))))

	cnt = 0
	waitMS = 40
	# loop over the unique components
	for (i, label) in enumerate(np.unique(labels)):
		# if this is the background label, ignore it
		if label == 0:
			#print("[INFO] label: 0 (background)")
			continue
 
		# otherwise, construct the label mask to display only connected components for
		# the current label
		#print("[INFO] label: {} (foreground)".format(i))
		labelMask = np.zeros(img.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
 
		# if the number of pixels in the component is sufficiently large, add it to our
		# mask of "large" blobs
		if numPixels > 200 and numPixels < 3000:
			mask = cv2.add(mask, labelMask)
			cnt = cnt + 1
			if (visualize == True):
				key = showResizeImg(mask, 'masking', waitMS, 0, 0)
				if key == ord('s'):
					waitMS = 500
				if key == ord('q'):
					waitMS = 100

	print('total blobs = {}'.format(cnt))
	if (visualize == True):
		img = cv2.bitwise_and(img, img, mask=mask)
		showResizeImg(img, 'extracted', 0, 0, 0)
	return img


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--path", required=True, help="Path to the image")
	ap.add_argument("-t", "--threshold", required=False, default='adaptive', help='threshold method')
	ap.add_argument("-v", "--visualize", required=False, default=False, action='store_true', help='visualize the intermediate process')
	args = vars(ap.parse_args())

	imgReader = ImageReader(args["path"])

	while True:
		cv2.destroyAllWindows()
		ret, img, imgName = imgReader.read()
		if (ret == False):
			break
		(h,w,_) = img.shape

		key = showResizeImg(img, imgName, 0, 0, 0)
		if key == ord('q'):
			break
		if (key == ord('p')):
			imgReader.previous()
			continue

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		# blur the image to remove noise
		blur = blur_img(gray, 'bilateral')

		# equalize the image to have better contrast
		blur = cv2.equalizeHist(blur)

		# threshold image to binary
		thresholded = threshold_img(blur, args['threshold'])

		# do some morphological operation to remove noise and have better shape
		morphologied = morphological_process(thresholded, 'closing', (1,1))
		
		# extract_blobs is really slow
		#morphologied = extract_blobs(morphologied)
		
		# find contours
		contour = cv2.findContours(morphologied, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contour = contour[1]

		# filter out some contours based on the prior knowledge about bib
		candidateList = filter_contours(img, contour, args['visualize'])

		# clear all intermediate images
		cv2.destroyAllWindows()

		draw_result(img, candidateList)
		
		key = cv2.waitKey(0)
		if key == ord('q'):
			break
		if (key == ord('p')):
			imgReader.previous()

main()