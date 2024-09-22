import cv2
import dlib
import time
import numpy as np

# Load the SVM model for object detection
model = dlib.simple_object_detector('../resources/my_detection_model.svm')

def resize_with_aspect_ratio(image, target_width=288, target_height=216, color=(0, 0, 0)):
    # Get original dimensions
    h, w = image.shape[:2]

    # Compute scaling factors to maintain aspect ratio
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)

    # Calculate new size while keeping the aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image to fit within the target dimensions
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new image with the target size and fill with the background color
    result = np.full((target_height, target_width, 3), color, dtype=np.uint8)

    # Calculate top-left corner to place the resized image
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    # Place the resized image on the center of the background
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return result

def resize_image(input_image, target_width=427, target_height=240):
    # Check if the input image is not empty
    if input_image is None or input_image.size == 0:
        print("Error: Empty input image!")
        return np.array([])  # Return an empty image in case of an error

    # Resize the input image to the target size using bilinear interpolation
    resized_image = cv2.resize(input_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return resized_image

class Detector:
    def detect_on_webcam(self, output_video_path):
        # Use webcam instead of video file (index 0 is the default camera)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print(f"Error: Unable to open webcam")
            return

        # Get webcam properties
        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object for saving output
        target_width, target_height = 288, 216  # Make sure this matches your resizing logic

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = resize_with_aspect_ratio(frame) # Resize

            # Measure object detection time per frame
            start_time = time.time()
            detected_objects = model(frame)
            end_time = time.time()

            detection_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"Detection time: {detection_time_ms:.2f} ms")

            # Draw bounding boxes and annotate detected objects
            for d in detected_objects:
                l, t, r, b = (
                    int(d.left()),
                    int(d.top()),
                    int(d.right()),
                    int(d.bottom())
                )

                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)
                # cv2.rectangle(frame, (l, b - 20), (r, b), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, 'ball', (l, b+12 ), cv2.FONT_HERSHEY_DUPLEX, ((r-l)/100)*1.5, (255, 255, 255), 1)

            # Write annotated frame to the output video
            out.write(frame)

            # Display frame with detections
            cv2.imshow('Custom Objects', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Detection video saved successfully at {output_video_path}")

if __name__ == '__main__':
    output_video_path = '../images/output_webcam_detect.mp4' # Specify your output video path here
    Detector().detect_on_webcam(output_video_path)
