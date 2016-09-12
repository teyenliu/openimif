from core.imif_digits import *
import cv2
import mahotas
import sys

# Initializes our image identifier class
imif = imif_digits()

# Trains and saves model
# imid.train_and_save_model('data/MNIST_digits', '../trained_models/mnist_digits.ckpt')

# Loads saved model
imif.load_model('trained_models/mnist_digits.ckpt')

# If we have at least one arguement, it's an image not video capture
if (len(sys.argv) > 1):
    image = cv2.imread(sys.argv[1])
    im_cpy = image.copy()

    # Converts to grayscale (to get contours)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Optimize for thresholding
    # imgray = cv2.bilateralFilter(imgray, 9, 90, 16)
    # imgray = cv2.medianBlur(imgray, 5)

    # thresholding
    # # thres = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
    # ret, thres = cv2.threshold(imgray, 80, 255, cv2.THRESH_BINARY)
    # thres = cv2.medianBlur(thres, 5)

    # blur the image, find edges, and then find contours along
    # the edged regions
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort the contours by their x-axis position, ensuring
    # that we read the numbers from left to right
    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key=lambda x: x[1])

    # loop over the contours
    for (c, _) in cnts:
        # compute the bounding box for the rectangle
        (x, y, w, h) = cv2.boundingRect(c)

        # if the width is at least 7 pixels and the height
        # is at least 20 pixels, the contour is likely a digit
        if w >= 1 and h >= 20:
            # crop the ROI and then threshold the grayscale
            # ROI to reveal the digit
                roi = gray[y:y + h, x:x + w]

                id_digit = imif.identify(roi)
                
                # draw a rectangle around the digit, the show what the
                # digit was classified as
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(image, str(id_digit), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    #cv2.imshow('blurred', blurred)
    #cv2.imshow('edged', edged)
    cv2.imshow('original', im_cpy)
    cv2.imshow('image extracted', image)

    cv2.waitKey()
    exit(0)

'''
cap = cv2.VideoCapture(-1)

while(True):
    # Gets frame
    ret, im = cap.read()

    # Converts to grayscale (to get contours)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Optimize for thresholding
    imgray = cv2.bilateralFilter(imgray, 9, 90, 16)
    imgray = cv2.medianBlur(imgray,5)

    # thresholding
    #thres = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
    ret, thres = cv2.threshold(imgray, 120, 255, cv2.THRESH_BINARY_INV)

    new_im, contours, hierarchy = cv2.findContours(thres.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cur = 0
    # Loop through each contours
    for cnt in contours:
        # Get moments of contour (to get location)
        M = cv2.moments(cnt)

        # Get area and location of contour
        if(M['m10'] != 0 and M['m00'] != 0 and M['m01'] != 0):
            area = cv2.contourArea(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            x, y, w, h = cv2.boundingRect(cnt)

            if (area > 250):
                try:
                    cur = cur + 1
                    #img_cpy = thres.copy()
                    cropped_img = thres[y-(h/4):y+h+h/3, x-(w/4):x+w+w/3].copy()
                    cropped_img = cv2.resize(cropped_img, (28, 28))
                    id_digit = imid.identify(cropped_img)
                    #cv2.imshow(str(cur), cropped_img)
                    cv2.rectangle(im, (x-(w/4), y-(h/4)), (x+w+w/3, y+h+h/3), (0, 255, 0), 2)
                    cv2.putText(im , str(id_digit), (x,y-y/4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                except:
                    pass


    #cv2.drawContours(im, contours, -1, (255, 0, 0), 3)
    cv2.imshow('image', im)
    cv2.imshow('thresholded', thres)

    key_pressed = cv2.waitKey(1) & 0xFF
    if (key_pressed == ord('q')):
        break

    if (key_pressed == ord('s')):
        cv2.imwrite('image.png', thres)

# Release capture
cap.release()
cv2.destroyAllWindows()
'''
