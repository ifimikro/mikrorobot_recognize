import face_recognition as frec
import cv2
import numpy as np
import json

# Reference to webcam
video_capture = cv2.VideoCapture(0)

class Face_Recog:
    def __init__(self):
        self.known_faces = self.load_known_faces()
        self.face_buffer = []
        self.face_i = 0
        # Starting main loop
        self.face_rec_loop()
        self.close_release_quit()

    def face_rec_loop(self):
        process_this_frame = True
        while True:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            if process_this_frame:
                face_locations = frec.face_locations(small_frame)
                if face_locations:
                    face_encodings = frec.face_encodings(small_frame, face_locations)
                    self.name_faces(face_locations, face_encodings, frame)
                # Show frame
                cv2.imshow('Video', frame)
            # Limit processing
            process_this_frame = not process_this_frame
            # To exit readloop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def add_face(self, faces):
        for i, face in enumerate(faces):
            if i != 0:
                new_face += face
            else:
                new_face = face
        new_face /= len(faces)
        obj = {}
        obj['name'] = input('What is the name of the person?\n-> ')
        if obj['name'] == 'n':
            return
        obj['encoding'] = new_face.tolist()
        self.known_faces.append(obj)
        self.save_face_to_file(self.known_faces)

    def name_faces(self, face_locations, face_encodings, frame):
        found = False
        for (top, right, bottom, left), face_enc in zip(face_locations, face_encodings):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            if len(self.known_faces) > 0:
                for known in self.known_faces:
                    if frec.compare_faces([known['encoding']], face_enc)[0]:
                        cv2.putText(frame, known['name'], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                        found = True
                        break
            else:
                cv2.putText(frame, 'Uknown', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if not found:
                cv2.putText(frame, 'Uknown', (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                self.get_to_know(face_enc)
            found = False

    def get_to_know(self, face_encoding):
        if len(self.face_buffer) == 0:
            self.face_buffer.append(face_encoding)
        elif len(self.face_buffer) > 0:
            if frec.compare_faces([self.face_buffer[self.face_i]], face_encoding):
                self.face_buffer.append(face_encoding)
                self.face_i += 1
            else:
                del self.face_buffer[:]
        if len(self.face_buffer) == 10:
            self.add_face(self.face_buffer)
            del self.face_buffer[:]
            self.face_i = 0

    def load_known_faces(self):
        with open('faces.json') as json_data:
            return json.load(json_data)

    def save_face_to_file(self, json_face):
        with open("faces.json", "w") as outfile:
            json.dump(json_face, outfile, indent=4)

    def close_release_quit(self):
        video_capture.release()
        cv2.destroyAllWindows()

Face_Recog()
