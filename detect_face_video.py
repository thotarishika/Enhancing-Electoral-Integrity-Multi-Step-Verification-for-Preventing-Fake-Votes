import cv2
import dlib
import numpy as np

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # make sure you have this file
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')  # make sure you have this file

# Convert the reference image to dlib's format
reference_img = dlib.load_rgb_image('test.jpg')  # replace with correct path
reference_faces = detector(reference_img, 1)
reference_shape = sp(reference_img, reference_faces[0])
reference_descriptor = np.array(facerec.compute_face_descriptor(reference_img, reference_shape))

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = detector(frame_rgb, 1)

    for face in faces:
        shape = sp(frame_rgb, face)
        descriptor = np.array(facerec.compute_face_descriptor(frame_rgb, shape))

        distance = np.linalg.norm(reference_descriptor - descriptor)
        if distance < 0.6:  # You might need to adjust this threshold
            # Draw a green rectangle with 'Match Found'
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, "Match Found", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Draw a red rectangle with 'No Match'
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            cv2.putText(frame, "No Match", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
