import sys                      # System bindings
import cv2                      # OpenCV bindings
import numpy as np
from collections import Counter
import urllib
import json
from PIL import Image # $ pip install pillow
 
class BackgroundColorDetector():
    def __init__(self, imageLoc):
        self.img = imageLoc
        self.manual_count = {}
        self.h, self.w, self.channels = self.img.shape
        self.total_pixels = self.w*self.h
 
    def count(self):
        for y in xrange(0, self.h):
            for x in xrange(0, self.w):
                RGB = (self.img[y,x,2],self.img[y,x,1],self.img[y,x,0])
                if RGB in self.manual_count:
                    self.manual_count[RGB] += 1
                else:
                    self.manual_count[RGB] = 1
 
    def average_colour(self):
        red = 0; green = 0; blue = 0;
        sample = 10
        for top in xrange(0, sample):
            red += self.number_counter[top][0][0]
            green += self.number_counter[top][0][1]
            blue += self.number_counter[top][0][2]
 
        average_red = red / sample
        average_green = green / sample
        average_blue = blue / sample
        return str(average_red)+", "+str(average_green)+", "+str(average_blue);
 
    def twenty_most_common(self):
        self.count()
        self.number_counter = Counter(self.manual_count).most_common(10)
        # for rgb, value in self.number_counter:
        #     print rgb, value, ((float(value)/self.total_pixels)*100)
 
    def detect(self):
        self.twenty_most_common()
        self.percentage_of_first = (float(self.number_counter[0][1])/self.total_pixels)
        # print self.percentage_of_first
        if self.percentage_of_first > 0.4:
            # print "Major Color"
            return str(self.number_counter[0][0][0])+","+str(self.number_counter[0][0][1])+","+str(self.number_counter[0][0][2])
        else:
            # print "Average Color"
            return str(self.average_colour())

def getImageColors(imgurl):
    imageDict = {}
    imageDict["imageUrl"] = imgurl
    im = Image.open(urllib.urlopen(imgurl))
    imcv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    y,x,z = imcv.shape
    # print imcv.shape
    ten_percent_x = x/10
    ten_percent_y = y/10
    crop_img_upper = imcv[0:ten_percent_y]
    # print crop_img_upper.shape
    crop_img_lower = imcv[y - ten_percent_y:y]
    # print crop_img_lower.shape
    # cv2.imshow('dst_upper',crop_img_upper)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    # cv2.imshow('dst_lower',crop_img_lower)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    BackgroundColorUpper = BackgroundColorDetector(crop_img_upper)
    topRgb = BackgroundColorUpper.detect()
    imageDict["topColor"] = topRgb
    BackgroundColorLower = BackgroundColorDetector(crop_img_lower)
    bottomRgb = BackgroundColorLower.detect()
    imageDict["bottomColor"] = bottomRgb
    return json.dumps(imageDict)

# print getImageColors(sys.argv[1])

# if __name__ == "__main__":
#     if (len(sys.argv) != 2):                        # Checks if image was given as cli argument
#         print "error: syntax is 'python main.py /example/image/location.jpg'";
#     else:
#         im = Image.open(urllib.urlopen(sys.argv[1]))
#         imcv = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
#         y,x,z = imcv.shape
#         ten_percent_x = x/10
#         ten_percent_y = y/10
#         crop_img_upper = imcv[0:y+ten_percent_y, 0:x+ten_percent_x]
#         crop_img_lower = imcv[y - ten_percent_y:y, x - ten_percent_x : x]
#         BackgroundColor = BackgroundColorDetector(crop_img_upper)
#         BackgroundColor.detect()
        # BackgroundColor = BackgroundColorDetector(crop_img_lower)
        # BackgroundColor.detect()