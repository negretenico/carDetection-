import cv2 as cv

#trained xml file for cars
car = cv.CascadeClassifier("C:/Users/Nico/AppData/Local/Programs/Python/Python37/Scripts/cars.xml")


#loads vidoe into cap
cap = cv.VideoCapture('C:/Users/Nico/AppData/Local/Programs/Python/Python37/Scripts/video.avb ni')


while True:

    #captures image
    read, img = cap.read()

    #turns image into a grayScale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    #detects cars
    cars = car.detectMultiScale(gray,1.1,5)
    for (x,y,w,h) in cars:
        #draws rectange on the image starting at the top left corner and drawing to the bottom right corner
        #the color will be blue with a thickness of 2
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display an image in a window
    cv.imshow('Window', img)

    # Wait for Esc key to stop
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv.destroyAllWindows()