import face_recognition as frec
import cv2
import numpy as np
import json

# Reference to webcam
video_capture = cv2.VideoCapture(0)

class Face_Recog:
    def __init__(self, gtk_mode):
        self.gtk_mode = gtk_mode
        self.known_faces = self.load_known_faces()

        # Starting main loop
        self.face_rec_loop()

        self.close_release_quit()

    def face_rec_loop(self):
        face_locations = []
        face_encodings = []
        face_names = []
        face_buffer = []
        face_i = 0
        process_this_frame = True
        while True:
            ret, frame = video_capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            if process_this_frame:
                face_locations = frec.face_locations(small_frame)
                face_encodings = frec.face_encodings(small_frame, face_locations)
                self.find_faces(face_locations, frame)
                if get_to_know_mode:
                    if len(face_buffer) == 0 and np.shape(face_encodings)[0] == 1:
                        face_buffer.append(face_encodings[0])
                    if len(face_buffer) > 0 and len(face_encodings) == 1:
                        if frec.compare_faces([face_buffer[face_i]], face_encodings[0]):
                            face_buffer.append(face_encodings[0])
                            face_i += 1
                    if len(face_buffer) == 5:
                        add_face(face_buffer)
                        del face_buffer[:]
                        face_i = 0

            # Show frame
            cv2.imshow('Video', frame)
            # Limit processing
            process_this_frame = not process_this_frame
            # To exit readloop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    def load_known_faces(self):
        with open('faces.json') as json_data:
            return json.load(json_data)

    def save_face_to_file(self, json_face):
        with open("faces.json", "w") as outfile:
            json.dump(json_face, outfile, indent=4)

    def add_face(self, faces):
        new_face = (faces[0] + faces[1] + faces[2] + faces[3] + faces[4]) / 5
        cv2.imshow('face', new_face)
        name = input('What is the name of the person?\n-> ')
        save_face_to_file({'name': name,'encoding': new_face.tolist()})

    def find_faces(self, face_locations, frame):
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    def close_release_quit(self):
        video_capture.release()
        cv2.destroyAllWindows()

    '''
    def find_known_faces(self, face_locations, frame):
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    '''
GTK_MODE = True
Face_Recog(GTK_MODE)
