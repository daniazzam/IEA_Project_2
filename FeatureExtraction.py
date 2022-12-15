import csv
from skimage.feature import hog
from GetBoundingRectange import *
from tqdm import tqdm
import pandas as pd


def getBlackToWhiteRatio(image):
    blackToWhiteRatio = np.sum(image == 0) / (np.sum(image == 255) + 0.00000000000000001)
    return blackToWhiteRatio



# Unused Features
'''
def getBlackRatio(image):
    blackRatio = np.sum(image==0)/(np.sum(image == 0) +np.sum(image == 255))
    return blackRatio

def horizontalSymmetry(image):
    blackTop = np.sum(image[0:15, :] == 0)
    blackBottom = np.sum(image[15:30, :] == 0)
    ratio = blackTop / blackBottom
    if 0.8 <= ratio <= 1.2:
        horizontal_Symmetry = 1
    else:
        horizontal_Symmetry = 0

    return horizontal_Symmetry


def inverseSymmetry(image):

    blackTopLeft = np.sum(image[0:15, 0:15] == 0)
    blackTopRight = np.sum(image[0:15, 15:30] == 0)
    blackBottomLeft = np.sum(image[15:30, 0:15] == 0) + 0.000000000000001
    blackBottomRight = np.sum(image[15:30, 15:30] == 0) + 0.000000000000001
    ratio1 = blackTopLeft / blackBottomRight
    ratio2 = blackTopRight / blackBottomLeft

    inverse_Symmetry = 0
    if 0.8 <= ratio1 and ratio2 <= 1.2:
        inverse_Symmetry = 1

    return inverse_Symmetry


def verticalSymmetry(image):
    blackLeft = np.sum(image[:, 0:15] == 0)
    blackRight = np.sum(image[:, 15:30] == 0) + 0.0000000000000000001
    ratio = blackLeft / blackRight
    if 0.9 <= ratio <= 1.1:
        vertical_Symmetry = 1
    else:
        vertical_Symmetry = 0

    return vertical_Symmetry

'''
def getAspectRatio(image):
    w, h = image.shape
    aspectRatio = h / w

    return aspectRatio


def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    normalized_matrix = matrix / norm

    return normalized_matrix


def getProjectionHistogram(image):
    image = cv2.bitwise_not(image)
    column_sum = np.sum(image, axis=0)  # sum the values in each column of the img
    row_sum = np.sum(image, axis=1)  # sum the values in each row of the img

    column_sum = column_sum /(28*255)
    row_sum = row_sum /(28*255)


    return column_sum, row_sum


def getProfile(image):
    image = cv2.bitwise_not(image)
    dimensions = image.shape  # (m,n) m:rows/ n:columns

    bottom = []
    bottom_sum = 0

    for j in range(dimensions[1]):  # columns
        for i in reversed(range(dimensions[0])):  # rows
            if image[i][j] == 0:
                bottom_sum = bottom_sum + 1
            elif image[i][j] == 255:
                break
        bottom = np.append(bottom, bottom_sum)
        bottom_sum = 0
    if np.sum(bottom) != 0:
        bottom = normalize_2d(bottom)
    """
            #####   TOP  #####
    """
    top = []
    top_sum = 0

    for j in range(dimensions[1]):  # columns
        for i in range(dimensions[0]):  # rows
            if image[i][j] == 0:
                top_sum = top_sum + 1
            elif image[i][j] == 255:
                break
        top = np.append(top, top_sum)
        top_sum = 0
    if np.sum(top) != 0:
        top = normalize_2d(top)

    """
            #####  RIGHT  #####
    """
    right = []
    right_sum = 0

    for i in range(dimensions[0]):  # rows
        for j in reversed(range(dimensions[1])):  # columns
            if image[i][j] == 0:
                right_sum = right_sum + 1
            elif image[i][j] == 255:
                break
        right = np.append(right, right_sum)
        right_sum = 0
    if np.sum(right) != 0:
        right = normalize_2d(right)

    """
            #####  left #####
    """
    left = []
    left_sum = 0

    for i in range(dimensions[0]):  # rows
        for j in range(dimensions[1]):  # columns
            if image[i][j] == 0:
                left_sum = left_sum + 1
            elif image[i][j] == 255:
                break
        left = np.append(left, left_sum)
        left_sum = 0
    if np.sum(left) != 0:
        left = normalize_2d(left)

    return bottom, top, left, right


def getHOG(image):
    resized_img = cv2.resize(image, (128, 64))
    hog_feature, image_hog = hog(resized_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4),
                                 visualize=True)  # Set visualize to true if we need to see the image
    return hog_feature

def crossings(image):
    white_row = np.full((30), 5)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.vstack([image, np.full((28), 255)])
    image = np.vstack([np.full((28), 255), image])
    column = np.ones((30, 1)) * 255
    image = np.hstack((column, image))
    image = np.hstack((image, column))

    for idx, row in enumerate(image):
        for idx2, elt in enumerate(row):
            if elt > 0:
                row[idx2] = 1
        image[idx] = row

    column_crossings = []
    for r in np.rollaxis(image, 1):
        original = 1
        number_of_crossings = 0
        for idx, elt in enumerate(r):
            if idx == 0:
                continue
            if elt != original:
                number_of_crossings += 1
                original = 0 if original == 1 else 1
        column_crossings.append(number_of_crossings)
        number_of_crossings = 0
        original = 1


    row_crossings = []
    for r in np.rollaxis(image, 0):
        original = 1
        number_of_crossings = 0
        for idx, elt in enumerate(r):
            if idx == 0:
                continue
            if elt != original:
                number_of_crossings += 1
                original = 0 if original == 1 else 1
        row_crossings.append(number_of_crossings)
        number_of_crossings = 0
        original = 1

    return column_crossings, row_crossings


def getListOfCharacters():
    listOfCharacters = []
    excelFile = 'EvalWhite.csv'
    with open(excelFile, 'r') as data:
        for line in tqdm(csv.DictReader(data)):
            listOfCharacters.append(line)

    return listOfCharacters


def featuresToCSV(listOfCharacters):  # directory of the cropped images
    dirname, filename = os.path.split(os.path.abspath(__file__))

    features = []

    for idx, img in tqdm(enumerate(listOfCharacters)):
        image_name = listOfCharacters[idx]['image']  # image name
        image_name2 = image_name[find_nth(image_name, '/', 1) + 1:]
        label = listOfCharacters[idx]['label']
        # small = 'abcdefghijklmnopqrstuvwxyz'

        # if label in small:
        #     label = f'_{label}'

        dirResizedWhite = dirname + '/' + image_name
        dirBounding = dirname + f'/BoundingBoxes/{image_name2}'

        boundingRectangleImage = cv2.imread(dirBounding)
        boundingRectangleImage = cv2.cvtColor(boundingRectangleImage, cv2.COLOR_BGR2GRAY)

        ResizedBoundingWithWhite = cv2.imread(dirResizedWhite)
        ResizedBoundingWithWhite = cv2.cvtColor(ResizedBoundingWithWhite, cv2.COLOR_BGR2GRAY)

        img['BlackToWhite'] = getBlackToWhiteRatio(boundingRectangleImage)
        # img['Horizontal Symmetry'] = horizontalSymmetry(boundingRectangleImage)
        # img['Inverse Symmetry'] = inverseSymmetry(boundingRectangleImage)
        # img['Vertical Symmetry'] = verticalSymmetry(boundingRectangleImage)
        img['Aspect Ratio'] = getAspectRatio(boundingRectangleImage)

        # Projection histogram
        column_sum, row_sum = getProjectionHistogram(ResizedBoundingWithWhite)
        for idx1, value in enumerate(column_sum):
            img[f'Column Histogram {idx1}'] = value

        for idx1, value in enumerate(row_sum):
            img[f'Row Histogram {idx1}'] = value

        # Profile
        bottom, top, left, right = getProfile(ResizedBoundingWithWhite)
        for idx1, value in enumerate(bottom):
            img[f'Bottom Profile {idx1}'] = value

        for idx1, value in enumerate(top):
            img[f'Top Profile {idx1}'] = value

        for idx1, value in enumerate(left):
            img[f'Left Profile {idx1}'] = value

        for idx1, value in enumerate(right):
            img[f'Right Profile {idx1}'] = value

        # Hog
        image_hog = getHOG(boundingRectangleImage)
        for idx1, value in enumerate(image_hog):
            img[f'HOG {idx1}'] = value

        col, row = crossings(ResizedBoundingWithWhite)
        for idx1, value in enumerate(col):
            img[f'Column Crossings {idx1}'] = value
        for idx1, value in enumerate(row):
            img[f'Row Crossings {idx1}'] = value

        # counter = 0
        # for row in ResizedBoundingWithWhite:
        #     for el in row:
        #         if el > 0:
        #             el = 255
        #         img[f'Pixel {counter}'] = el
        #         counter += 1

        features.append(img)

    df = pd.DataFrame(features)
    df.to_csv('EvalWhiteFeatures.csv', index=False, header=True)


def saveToCSV():
    list_of_Characters = getListOfCharacters()
    featuresToCSV(list_of_Characters)


if __name__ == '__main__':
    saveToCSV()