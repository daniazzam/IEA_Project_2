import cv2
import numpy as np
import os
import csv
from tqdm import tqdm

"""
A function to get the binary black and white image from a RGB image
"""

def BGR2BINARY (image, x,y):

    # convert to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply the threshold

    blur = cv2.GaussianBlur(gray_image,(5,5),0)
    _, thresh_image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Negate the image to get a black background and white character
    binary_image = cv2.bitwise_not(thresh_image)
    # Apply opening to remove noise
    kernel = np.ones((x,y),np.uint8)
    final_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return final_image


"""
A function to get the bounding rectange of the binary image
"""


def getBoundingRect(image):

    x1,y1,w,h = cv2.boundingRect(image)
    
    x2 = x1+w
    y2 = y1+h
    bounding_rect_image = image [y1:y2,x1:x2]

    #Return image to white background with 

    bounding_rect_image = cv2.bitwise_not(bounding_rect_image)
    return bounding_rect_image


def getCountourRect(image):
    countours, hir = cv2.findContours(image, 1, cv2.CHAIN_APPROX_SIMPLE)
    # print('number of count: ' + str(len(countours)))
    if len(countours) > 1:
        countours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h = 0
        img = 0
        for c in countours:
            x, y, w, h_temp = cv2.boundingRect(c)
            if h_temp > h:
                h = h_temp
                img = c
        x, y, w, h = cv2.boundingRect(img)
        x2 = x + w
        y2 = y + h
        bounding_rect_image = image[y:y2, x:x2]
        bounding_rect_image = cv2.bitwise_not(bounding_rect_image)
        return bounding_rect_image
    else:
        return getBoundingRect(image)


"""
A function to get the bounding box picture resized to a square
"""


def resizeToSquare(boundingBox):
    box_height, box_width = boundingBox.shape
    if box_height >= box_width:
        img = np.zeros((box_height, box_height, 3), dtype=np.uint8)
        img[:, :] = 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sizeDifference = int((box_height - box_width) / 2)
        img[:, sizeDifference:sizeDifference + box_width] = boundingBox
    else:
        img = np.zeros((box_width, box_width, 3), dtype=np.uint8)
        img[:, :] = 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sizeDifference = int((box_width - box_height) / 2)
        img[sizeDifference:sizeDifference + box_height, :] = boundingBox
    return img


"""
Method to resize square frame image
"""


def resizeImage(img, height, width):
    return cv2.resize(img, (height, width))


"""
Method that does all the previous steps in one function
"""


def processImage(dir):
    n = dir.find('img')
    srcImg = cv2.imread(dir)
    # print()
    if dir[n + 3:n + 6] == '045' or dir[n + 3:n + 6] == '046':
        # print('i or j')
        binaryImage = BGR2BINARY(srcImg, 2, 2)
        boundingRect = getBoundingRect(binaryImage)
    else:

        # print('not i neither j') 

        binaryImage = BGR2BINARY(srcImg, 2, 2)
        boundingRect = getCountourRect(binaryImage)

    return boundingRect


"""
Save images to directory
"""


def saveImages(dirName):
    for filename in os.listdir(dirName):
        f = os.path.join('./' + dirName, filename)
        if os.path.isfile(f):
            toAdd = processImage(f, 28, 28)
            cv2.imwrite(f, toAdd)


def getListOfCharacters():
    listOfCharacters = []
    excelFile = 'eval.csv'
    with open(excelFile, 'r') as data:
        for line in csv.DictReader(data):
            listOfCharacters.append(line)

    return listOfCharacters


"""
Main function
"""


def main():
    directory = 'by_class'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if f != 'by_class/.DS_Store':
            for letter in os.listdir(f):
                ch = os.path.join(f, letter)


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start
        

if __name__ == '__main__':
    dirname, filename = os.path.split(os.path.abspath(__file__))
    list_of_Characters = getListOfCharacters()
    small = 'abcdefghijklmnopqrstuvwxyz'

    for idx, img in tqdm(enumerate(list_of_Characters)):
        image_name = list_of_Characters[idx]['image']  # image name
        image_name2 = image_name[find_nth(image_name, '/', 2) + 1:]
        label = list_of_Characters[idx]['label']
     
        fullPath = dirname + '/' + image_name
        if os.path.exists(fullPath):
            boundingRect = processImage(fullPath)
            # boundingRect = resizeToSquare(boundingRect)
            # boundingRect = cv2.resize(boundingRect, (28,28))
            if label in small:
                label = f'_{label}' 
            save_dir = dirname + f'/ANN_Eval_BR/{label}/{image_name2}'
            cv2.imwrite(save_dir, boundingRect)

