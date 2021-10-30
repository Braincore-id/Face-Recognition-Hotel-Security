import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
guest = {
	1: [
		'Eric Julianto',
		'Lantai 7',
		'R707', 
		'Twin Bed',
		'VIP Guest'
	],
	2: [
		'Eric Julianto', 
		'Lantai 7',
		'R707', 
		'Twin Bed',
		'VIP Guest'
	],
}

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
	ret, img =cam.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale( 
		gray,
		scaleFactor = 1.5,
		minNeighbors = 5,
		minSize = (int(minW), int(minH)),
	)

	for(x,y,w,h) in faces:
		id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
		# Check if confidence is less them 100 ==> "0" is perfect match 
		if (confidence < 65): # known pasenger and match > 35%
			confidence = "  {0}%".format(round(100 - confidence))
			try:
				datainfo = guest[id]
				cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2) # add green box

				tinggi = 5
				for i in datainfo:
					cv2.putText(img, i, (x+w+5,y+tinggi), font, 1, (255,255,255), 2)
					tinggi += 40
				cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
			except:
				# error load pasenger
				pass
		else:
			confidence = "  {0}%".format(round(100 - confidence))
			datainfo = ['unknown guest']
			# unkown passenger
			cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2) # add red box
			cv2.putText(img, datainfo[0], (x+w+5,y+5), font, 1, (255,255,255), 2)

			# confidence info
			cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

	cv2.imshow('Guest Details',img) 
	k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
	if k == 27:
		break

cam.release()
cv2.destroyAllWindows()
