
from __future__ import print_function

import numpy as np
import argparse
import cv2    
import csv
import os 

def threshold_by_color_filter(imgPath, img):

	# read the upper/lower color from csv file
	f = open(os.path.dirname(imgPath)+'/color.csv', 'rb')
	reader = csv.reader(f, delimiter=' ',quoting=csv.QUOTE_MINIMAL)
	lowerColor, upperColor = next(reader)

	lowerColor = lowerColor.translate(None, '[],').split()
	lowerColor = [int(v) for v in lowerColor]

	upperColor = upperColor.translate(None, '[],').split()
	upperColor = [int(v) for v in upperColor]


	#print('color = {}, {}'.format(lowerColor[0], upperColor[0]))
	f.close()	

	#img = cv2.imread(imgPath)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, np.array(lowerColor, dtype='uint8'), np.array(upperColor, dtype='uint8'))
	return mask