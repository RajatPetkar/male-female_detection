import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved gender classification model
gender_model = load_model('gender_classification_model.h5')

# Load the pre-trained person detection model (HOG + SVM)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Access the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, change if using a different camera

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    
    if not ret or frame is None:
        print("Error: No camera or empty frame")
        break

    # Perform person detection
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8))

    if len(boxes) == 0:
        # No person detected, display "Nothing found"
        cv2.putText(frame, "Nothing found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in boxes:
            # Draw bounding box around the detected person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract the detected person region for gender classification
            person_roi = frame[y:y + h, x:x + w]
            person_roi_resized = cv2.resize(person_roi, (150, 150))
            person_img_array = np.expand_dims(person_roi_resized, axis=0)
            person_img_array = person_img_array / 255.0
            
            # Perform gender classification on the detected person
            predictions = gender_model.predict(person_img_array)
            gender = "Female" if predictions[0] > 0.5 else "Male"
            
            # Display the predicted gender
            cv2.putText(frame, f"Predicted: {gender}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Gender and Person Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
