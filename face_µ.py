import face_recognition
import cv2
import numpy as np
import pyttsx3

'''
    Notes:
        - add server to get images from
        - work on robustness and speed
        - have a faster algorithm finding people, use this to remember faces
'''

# Reference to webcam
video_capture = cv2.VideoCapture(0)

# load sample and learn to recognize
obama_image = face_recognition.load_image_file("myself2.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

trump_image = face_recognition.load_image_file("trump.jpg")
trump_face_encoding = face_recognition.face_encodings(trump_image)[0]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# webcam readloop
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    if process_this_frame:
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces([obama_face_encoding, trump_face_encoding], face_encoding)
            name = "Unknown"

            if match[0]:
                name = "Vetle"
            elif match[1]:
                name = "trump"


            face_names.append(name)
    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # To exit readloop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
