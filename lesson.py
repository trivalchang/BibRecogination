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
from utility import ocr
from utility import basics
from utility.image_processing.four_point_transform import four_point_transform

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

def minWindth(w):
	return (w/3) 

def maxWindth(w):
	return (w*3) 

def minHeight(h):
	return (h*0.75)

def maxHeight(h):
	return (h*1.5)

def minArea():
	global imageW, imageH
	a = imageW * imageH
	a = a / (200*200)
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
	if (x > v_right):
		debug_print('x is not compatible: x={}, v_right = {}, vw = {}, '.format(x, v_right, vw))
		return True
	if (x < vx):
		debug_print('x is not compatible: x={}, vx = {} '.format(x, vx))
		return True


	return False	

def find_toRemoveList(img, subList, virtualize=True):

	if len(subList) == 1:
		return list(subList)

	voteList = []
	max_vote = None	
	debug_print('===============================================')
	clone = img.copy()
	color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	for ((x, y, w, h), cnt) in subList:
		area = cv2.contourArea(cnt)

		if (virtualize == True):
			cv2.drawContours(clone, [cnt], -1, (0, 0, 255), CONTOUR_WIDTH)
			showResizeImg(clone, '{}'.format((x, y, w, h)), 0, 0, 0)

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
		basics.showResizeImg(clone, 'original contour', 0, 0, 0)

	filtered = []
	filteredBox = []
	possibleCandidate = []
	for c in contours:
		clone = img.copy()
		area = cv2.contourArea(c)
		(x, y, w, h) = cv2.boundingRect(c)
		rectArea = w*h
		solidity = area/float(rectArea)
		ar = float(w)/float(h)
		#cv2.drawContours(clone, [c], -1, (255, 0, 0), CONTOUR_WIDTH)
		#basics.showResizeImg(clone, 'area={}, ar={}'.format(area, ar), 0, 0, 0)
		if (area < minArea()):
			continue
		else:

			if (ar > 0.2) and (ar < 3):
				color = (0, 255, 0)
				filtered.append(c)
				filteredBox.append((x, y, w, h))
			elif (ar > 0.2) and (ar < 3):
				possibleCandidate.append(((x, y, w, h), c))
			else:
				color = (0, 0, 255)

	if (virtualize == True):
		cv2.drawContours(clone, filtered, -1, (255, 0, 0), CONTOUR_WIDTH)
		basics.showResizeImg(clone, 'remove unlikely: keep blue', 0, 0, 0)

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
		basics.showResizeImg(clone, 'sorted by y position', 0, 0, 0)

	filtered = []
	clone = img.copy()
	finalCandidateList = []
	for subList in candidateList:
		toRemove = find_toRemoveList(clone, subList, virtualize=False)
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
			if ar > 1 and ar < 8:
				finalCandidateList.append(((x0, y0, x1, y1), subList))

	if (virtualize == True):
		basics.showResizeImg(clone, 'red will be removed', 0, 0, 0)	

	return (finalCandidateList, possibleCandidate)

def draw_result(img, imgName, bibList):

	debug_print('###################################')

	if len(bibList) == 0:
		basics.showResizeImg(img, imgName, 1, 0, 0)
		return

	clone = img.copy()
	(imageH,imaheW) = img.shape[:2]

	h_gap = 10
	v_gap = 10
	startY = 0	

	maxW = 0
	if len(bibList) != 0:
		maxW = np.amax([x1-x0 for ((x0, y0, x1, y1), _) in bibList])

	resultImg = np.zeros((imageH,imaheW+maxW+h_gap*2, 3), dtype=np.uint8)
	candidateImg = []
	candidateBoundingBox = []
	for ((x0, y0, x1, y1), text) in bibList:
		print('********************')

		cv2.putText(clone, text, (x0,y0-30), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,0),5,cv2.LINE_AA)			
		cv2.rectangle(clone, (x0, y0), (x1, y1), (255, 0, 0), CONTOUR_WIDTH)
		resultImg[startY:startY+y1-y0, imaheW+h_gap:imaheW+h_gap+x1-x0] = img[y0:y1, x0:x1]
		startY = startY + y1 - y0 + v_gap

	resultImg[0:imageH, 0:imageW] = clone
	basics.showResizeImg(resultImg, imgName, 1, 0, 0)
	return (candidateImg, candidateBoundingBox)

def recognizeBibNumber(img, imgName, candidateList):

	if len(candidateList) == 0:
		return

	BibList = []
	for ((x0, y0, x1, y1), subList) in candidateList:
		candidateImg = img[y0:y1, x0:x1]
		contour = cv2.findContours(candidateImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
		text = ''
		boundingBoxes = [cv2.boundingRect(c) for c in contour]
		(contour, boundingBoxes) = zip(*sorted(zip(contour, boundingBoxes), key=lambda b:b[1][0]))
		for (c, (x, y, w, h)) in zip(contour, boundingBoxes):
			if (w*h) < 50:
				continue

			cropImg = candidateImg[y:y+h, x:x+w]
			c0 = cv2.findContours(cropImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1][0]
			if (len(c0) == 0):
				continue
			rect = cv2.minAreaRect(c0)
			box = np.int0(cv2.boxPoints(rect))
			newImg = four_point_transform(cropImg, box)
			text = text + ocr.ocr(newImg, method='bbp')
		BibList.append(((x0, y0, x1, y1), text))
	return BibList

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
	candidateCnt = 0

	while True:
		cv2.destroyAllWindows()
		ret, img, imgName = imgReader.read()
		if (ret == False):
			break

		if (args["resize"] == True):
			img = basics.resizeImg(img)

		(imageH, imageW,_) = img.shape

		if (args['visualize'] == True):
			key = basics.showResizeImg(img, imgName, 0, 0, 0)
			if key == ord('q'):
				break
			if (key == ord('p')):
				imgReader.previous()
				continue

		if args['threshold'] != 'color':
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
			# blur the image to remove noise
			blur = basics.blur_img(gray, 'bilateral')

			# equalize the image to have better contrast
			blur = cv2.equalizeHist(blur)

			# threshold image to binary
			thresholded = basics.threshold_img(blur, args['threshold'])
			# do some morphological operation to remove noise and have better shape
			thresholded = basics.morphological_process(thresholded, 'closing', (1,1))

		else:
			mask = threshold_by_color_filter(imgName, img)

			if args["resize"] == True:
				ksize = (1, 1)
			else:
				ksize = (3, 3)
			mask = basics.morphological_process(mask, 'closing', ksize)
			
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			adaptive = basics.threshold_img(gray, 'adaptive')

			if args["resize"] == True:
				ksize = (5, 5)
			else:
				ksize = (13, 13)

			adaptive = basics.morphological_process(adaptive, 'closing', ksize)
			thresholded = cv2.bitwise_and(adaptive, adaptive, mask=mask)
			if (args['visualize'] == True):
				basics.showResizeImg(mask, 'mask', 0, 900, 0)
				basics.showResizeImg(adaptive, 'adaptive thresholded', 0, 900, 0)
				basics.showResizeImg(thresholded, 'bitwise_and', 0, 900, 0)

		# find contours
		contour = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour = contour[1]

		# filter out some contours based on the prior knowledge about bib
		(candidateList, possibleCandidate) = filter_contours(img, contour, args['visualize'])

		# clear all intermediate images
		cv2.destroyAllWindows()

		BibList = recognizeBibNumber(thresholded, imgName, candidateList)
		draw_result(img, imgName, BibList)


		key = cv2.waitKey(0)
		if key == ord('q'):
			break
		if (key == ord('p')):
			imgReader.previous()

main()