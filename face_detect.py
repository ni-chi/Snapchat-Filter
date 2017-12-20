import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

def hatt(f, x, y, w, h) :
	hat = cv2.imread('hat.png')
	hatw = w + 1
	hath = int(0.35*h) + 1
	hat = cv2.resize(hat,(hatw, hath))

	for i in range(hath) :
		for j in range(hatw) :
			for k in range(3) :
				if hat[i][j][k] < 235 :
					f[y + i - int(0.25*h)][x + j][k] = hat[i][j][k]

	return f

while True :
	ret,frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.3,minNeighbors = 5,minSize = (30, 30))

	for (x, y, w, h) in faces :
		# cv2.circle(frame, (x+(w/2), y+(h/2)), w/2, (0, 255, 0), 2)
		frame = hatt(frame, x, y, w, h)

	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q') :
		break

video_capture.release()
cv2.destroyAllWindows()