import cv2
from GetBoundingRectange import resizeToSquare
from GetBoundingRectange import resizeImage

def segment(image):
    # Read the input image
    img = cv2.imread(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours of the letters in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters = []
    # Iterate through the contours and separate the letters
    for c in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(c)

        # Extract the letter from the original image and resize it to a fixed size
        letter_img = gray[y:y + h, x:x + w]
        letter_img = resizeToSquare(letter_img)
        letter_img = resizeImage(letter_img, 28, 28)

        # Save the processed letter image
        letters.append(letter_img)
        # cv2.imwrite(f"letter_{x}.png", letter_img)
    return letters