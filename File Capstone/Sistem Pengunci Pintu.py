from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time
import cv2
import RPi.GPIO as GPIO

RELAY = 17
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY, GPIO.OUT)
GPIO.output(RELAY,GPIO.LOW)

currentname = "unknown"  # inisialisasi 'currentname' untuk mendeteksi hanya pengguna baru yang diidentifikasi.
encodingsP = "encodings.pickle"
cascade = "haarcascade_frontalface_default.xml"  # alamat xml : https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

# memuat wajah dan embeddings yang ada di dataset dengan Haar OpenCV
print("[INFO] loading encodings + mendetelsi wajah…")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# inisialisasi dengan video 
print("[INFO] starting video…")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
prevTime = 0
doorUnlock = False

# perulangan frame pada file video
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	
    # OpenCV mengembalikan koordinat kotak pembatas dalam urutan (x, y, w, h)
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# perulangan deteksi wajah
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"], # mencocokkan setiap wajah pada gambar input dengan dataset
			encoding)
		name = "Unknown"

		if True in matches:
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}
			
			# fungsi membuka kunci
			GPIO.output(RELAY,GPIO.LOW)
			prevTime = time.time()
			doorUnlock = True
			print("door unlock")
			
            # perulangan indeks yang cocok untuk setiap  wajah di dataset
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			name = max(counts, key=counts.get)
			
			if currentname != name: # saat wajah teridentifikasi, cetak nama mereka di layar
				currentname = name
				print(currentname)
				names.append(name) # memperbarui daftar nama
        
    #kunci pintu setelah 5 detik
	if doorUnlock == True and time.time() - prevTime > 5:
		doorUnlock = False
		GPIO.output(RELAY,GPIO.HIGH)
		print("door lock")

	# perulangan untuk wajah yang teridentifikasi pada dataset
	for ((top, right, bottom, left), name) in zip(boxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			.8, (255, 0, 0), 2)
	cv2.imshow("Facial Recognition is Running", frame) # display the image to our screen
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"): # keluar saat tombol 'q' ditekan
		break
	fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()