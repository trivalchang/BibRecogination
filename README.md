# BibRecogination

This program aims to locate the bib of athletes by some basic image processing techniques. However, the result is poor.

**Usage**

usage: lesson.py [-h] -p PATH [-t THRESHOLD] [-v] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to the image
  -t THRESHOLD, --threshold THRESHOLD
                        threshold method (adaptive, OTSU)
  -v, --visualize       visualize the intermediate process
  -d, --debug           output debug info
  
**Example**
```
python lesson.py -p sport -t adaptive -d -v
```
the above command reads the image under folder *sport* and draws a green rectangle around the bib in the image. 
  

**Environment**

Python 2.7

OpenCV 3.3

macOS Sierra 

