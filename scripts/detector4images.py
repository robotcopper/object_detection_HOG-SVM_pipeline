import cv2
import dlib

# Load the SVM model for object detection
model = dlib.simple_object_detector('../resources/my_detection_model.svm')

class Detector:
        
    def detect_on_image(self, image_path):
        # Load the image from the specified path
        frame = cv2.imread(image_path)

        # Detect objects in the image
        detected_objects = model(frame)

        # Draw bounding boxes and annotate detected objects
        for d in detected_objects:
            l, t, r, b = (
                int(d.left()),
                int(d.top()),
                int(d.right()),
                int(d.bottom())
            )

            cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
            # cv2.rectangle(frame, (l, b - 20), (r, b), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, 'ball', (l, b+12 ), cv2.FONT_HERSHEY_DUPLEX, ((r-l)/100)*1.5, (255, 255, 255), 1)

        # Display the image with detections
        cv2.imshow('Custom Objects', frame)
        # Loop to wait for 'q' key or window close event
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Check if the window is closed
            if cv2.getWindowProperty('Custom Objects', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = '../images/validation/001_resized.jpg'  # Specify your image path here
    Detector().detect_on_image(image_path)
