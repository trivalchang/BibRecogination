from __future__ import print_function

import numpy as np
import argparse
import cv2    
from scipy import signal
import os 
import sys
import random
from skimage import measure
from color_filter import threshold_by_color_filter

CONTOUR_WIDTH = 5

bEnableDebug = False
imageH = 0
imageW = 0

def debug_print(dbgStr):
	if bEnableDebug == True:
		print(dbgStr)

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

def resizeImg(img, width=1280, height=720):
	(h,w) = img.shape[:2]
	r = min([float(width)/w, float(height)/h])
	(w, h) = (int(r * w), int(r * h))

	img = cv2.resize(img, (w, h))
	return img

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
	elif (method == 'blackhat'):
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
		blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
		return blackhat
	elif (method == 'erode'):
		eroded = cv2.erode(img, None, iterations=1)
		return eroded
	elif (method == 'dilate'):
		dilated = cv2.dilate(eroded, None, iterations=1)
		return dilated
	return img.copy()

def minWindth(w):
	return (w/3) 

def maxWindth(w):
	return (w*1.75) 

def minHeight(h):
	return (h*0.75)

def maxHeight(h):
	return (h*1.5)

def minArea():
	global imageW, imageH
	a = imageW * imageH
	a = a / (100*100)
	return a

def filterUnwanted(candidate, filterCord):
	global bEnableDebug

	(x, y, w, h) = candidate
	(vx, vy, vw, vh, v_right, v_bottom) = filterCord
				
	if (w < minWindth(vw)) or (w > maxWindth(vw)):
		debug_print('w is not compatible: w={}, vw = {}'.format(w, vw))
		return True
	if (h < minHeight(vh)) or (h > maxHeight(vh)):
		debug_print('h is not compatible: h={}, vh = {}'.format(h, vh))
		return True
	if (y < vy) or (y > v_bottom):
		debug_print('y is not compatible: y={}, vy = {}, vh = {}, '.format(y, vy, vh))
		return True
	# assume x is sorted in ascending order 
	if (x > v_right):
		debug_print('x is not compatible: x={}, v_right = {}, vw = {}, '.format(x, v_right, vw))
		return True

	return False	

def find_toRemoveList(img, subList):

	if len(subList) == 1:
		return list(subList)

	voteList = []
	max_vote = None	
	debug_print('===============================================')
	color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	for ((x, y, w, h), cnt) in subList:
		area = cv2.contourArea(cnt)

		if len(voteList) == 0:
			debug_print('new vote = {}'.format((x, y, w, h, x+w, y+h, 1)))
			voteList.append(((x, y, w, h), x+w, y+h, 1))
			max_vote = voteList[0]
		else:
			found_v = False
			for v in voteList:
				((vx, vy, vw, vh), v_right, v_bottom, v_vote) = v
				if filterUnwanted((x, y, w, h), (vx, vy-vh*0.5, vw, vh, v_right + vw, v_bottom+vh/2)) == True:
					continue

				debug_print('examine {}'.format((x, y, w, h)))
				debug_print('		v_right = {} to {} '.format(v_right, x + w))
				v_right = (x + w)
				vy = min(y, vy)
				v_bottom = min(y+h, v_bottom)
				voteList[voteList.index(v)] = ((vx, vy, vw, vh), v_right, v_bottom, v_vote+1)
				debug_print('add vote {} '.format((vx, vy, vw, vh, v_right, v_bottom, v_vote+1)))
				(_, _, _, vote_cnt) = max_vote
				if (vote_cnt < (v_vote+1)):
					max_vote = ((vx, vy, vw, vh), v_right, v_bottom, v_vote+1)
				found_v = True
				break
			if (found_v == False):
				voteList.append(((x, y, w, h), x+w, y+h, 1))
				debug_print('new vote = {}'.format((x, y, w, h, x+w, y+h, 1)))

	toRemove = []
	((vx, vy, vw, vh), v_right, v_bottom, v_vote) = max_vote
	if (v_vote == 1):
		return list(subList)

	debug_print('-------------')
	debug_print('final vote = {}'.format((vx, vy, vw, vh, v_right, v_bottom, v_vote)))
	for ((x, y, w, h), cnt) in subList:
		if filterUnwanted((x, y, w, h), (vx, vy, vw, vh, v_right, v_bottom)) == True:			
			toRemove.append(((x, y, w, h), cnt))
			debug_print('removed {}'.format((x, y, w, h)))
			continue
		debug_print((x, y, w, h))	

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
		if (area < minArea()):
			continue
		else:

			(x, y, w, h) = cv2.boundingRect(c)
			rectArea = w*h
			solidity = area/float(rectArea)
			ar = float(w)/float(h)
			if (ar > 0.2) and (ar < 0.9):
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
	clone = img.copy()
	(lastX, lastY, lastW, lastH) = (0, 0, 0, 0)
	for ((x, y, w, h), cnt) in zip(filteredBox, filtered):
		if (y > (lastY+lastH/2)) or (y < (lastY-lastH/2)):
			if len(subList) > 0:
				subList = sorted(subList, key=lambda a:a[0], reverse=False)
				candidateList.append(list(subList))
			subList = []
		subList.append(((x, y, w, h), cnt))
		(lastX, lastY, lastW, lastH) = (x, y, w, h)

	if (len(subList) > 0):
		subList = sorted(subList, key=lambda a:a[0], reverse=False)
		candidateList.append(list(subList))

	clone = img.copy()
	if (virtualize == True):
		for subList in candidateList:
			color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
			for ((x, y, w, h), cnt) in subList:
				cv2.drawContours(clone, [cnt], -1, color, CONTOUR_WIDTH)
		showResizeImg(clone, 'sorted by y position', 0, 0, 0)

	filtered = []
	clone = img.copy()
	finalCandidateList = []
	for subList in candidateList:
		toRemove = find_toRemoveList(clone, subList)
		subList = [item for item in subList if item not in toRemove]
		if len(subList) > 1:
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
				finalCandidateList.append(((x0, y0, x1, y1), subList))

	if (virtualize == True):
		showResizeImg(clone, 'red will be removed', 0, 0, 0)	

	return finalCandidateList

def draw_result(img, candidateList):
	clone = img.copy()
	debug_print('###################################')
	idx = 0
	startY = 0
	for ((x0, y0, x1, y1), subList) in candidateList:
		print('********************')
		for ((x, y, w, h), cnt) in subList:
			print('{}'.format((x, y, w, h)))
			cv2.drawContours(clone, [cnt], -1, (0, 255, 0), CONTOUR_WIDTH)	
			
		cv2.rectangle(clone, (x0, y0), (x1, y1), (255, 0, 0), CONTOUR_WIDTH)
		bib = img[y0:y1, x0:x1]
		showResizeImg(bib, 'BIB{}'.format(idx), 1, 800, startY, x1-x0, y1-y0)

		idx = idx + 1
		startY = startY + y1 - y0 + 50

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

	global bEnableDebug
	global imageW, imageH

	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--path", required=True, help="Path to the image")
	ap.add_argument("-r", "--resize", required=False, default=False, action='store_true', help="Path to the image")
	ap.add_argument("-t", "--threshold", required=False, default='adaptive', help='threshold method')
	ap.add_argument("-v", "--visualize", required=False, default=False, action='store_true', help='visualize the intermediate process')
	ap.add_argument("-d", "--debug", required=False, default=False, action='store_true', help='output debug info')
	args = vars(ap.parse_args())

	bEnableDebug = args["debug"]

	imgReader = ImageReader(args["path"])

	while True:
		cv2.destroyAllWindows()
		ret, img, imgName = imgReader.read()
		if (ret == False):
			break

		if (args["resize"] == True):
			img = resizeImg(img)

		(imageH, imageW,_) = img.shape

		if (args['visualize'] == True):
			key = showResizeImg(img, imgName, 0, 0, 0)
			if key == ord('q'):
				break
			if (key == ord('p')):
				imgReader.previous()
				continue

		if args['threshold'] != 'color':
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
			# blur the image to remove noise
			blur = blur_img(gray, 'bilateral')

			# equalize the image to have better contrast
			blur = cv2.equalizeHist(blur)

			# threshold image to binary
			thresholded = threshold_img(blur, args['threshold'])
			# do some morphological operation to remove noise and have better shape
			thresholded = morphological_process(thresholded, 'closing', (1,1))

		else:
			mask = threshold_by_color_filter(imgName, img)

			if args["resize"] == True:
				ksize = (1, 1)
			else:
				ksize = (3, 3)
			mask = morphological_process(mask, 'closing', ksize)
			
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			adaptive = threshold_img(gray, 'adaptive')

			if args["resize"] == True:
				ksize = (5, 5)
			else:
				ksize = (13, 13)

			adaptive = morphological_process(adaptive, 'closing', ksize)
			thresholded = cv2.bitwise_and(adaptive, adaptive, mask=mask)
			
			if (args['visualize'] == True):
				showResizeImg(mask, 'mask', 0, 900, 0)
				showResizeImg(thresholded, 'adaptive thresholded', 0, 900, 0)
				showResizeImg(thresholded, 'bitwise_and', 0, 900, 0)

		# find contours
		contour = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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