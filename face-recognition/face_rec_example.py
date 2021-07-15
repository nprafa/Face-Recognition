import cv2
import os
import face_recognition


TRAIN_DIR = "archive//data//data"
TEST_DIR = "archive//test//test"

TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"


print("loading train data")

known_faces = []
known_names = []

for name in os.listdir(TRAIN_DIR):
    for filename in os.listdir(f"{TRAIN_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{TRAIN_DIR}/{name}/{filename}")
        encodings = face_recognition.face_encodings(image)[0]
        known_faces.append(encodings)
        known_names.append(name)


print("loading test data")

for filename in os.listdir(f"{TEST_DIR}"):
    image = face_recognition.load_image_file(f"{TEST_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_locations in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        if True in results:
            match = known_names[results.index(True)]
            print(f"match found:{match}")
            top_left = (face_locations[3], face_locations[0])
            bottom_right = (face_locations[1], face_locations[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            top_left = (face_locations[3], face_locations[2])
            bottom_right = (face_locations[1], face_locations[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_locations[3] + 10, face_locations[2] + 15), cv2.FONT_ITALIC, 0.5, (200,200,200), FONT_THICKNESS)
        cv2.imshow(filename, image)
        cv2.waitKey(10000)
        cv2.destroyWindow(filename)








