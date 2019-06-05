import cv2
import os

# webcam
cap = cv2.VideoCapture(0)

# output file
outputFile = "video/name_of_the_video.mp4"

if os.path.exists(outputFile):
    i = ""
    while i != "n" and i != "y":
        i = input("Do yuo want to overwrite file '" + outputFile + "'? (y/n)\n")
    if i == "n":
        exit(0)

''' cycle begins '''
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(*'XVID'), 15, (width, height))
show = True
scale = 2
while True:

    ''' handle input '''
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord(' '):
        show = not show
    if not show:
        continue

    ''' reading next frame '''
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    out.write(frame)

cap.release()
cv2.destroyAllWindows()
out.release()