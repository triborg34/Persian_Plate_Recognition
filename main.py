from ultralytics import YOLO
import cv2
import math
import time
import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Get the current timestamp for output names
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
file_name = f'output_{timestamp}.jpg'
classnames = ['car', 'plate']
charclassnames = ['0', '9', 'b', 'd', 'ein', 'ein', 'g', 'gh', 'h', 'n', 's', '1', 'malul', 'n', 's', 'sad', 't', 'ta', 'v', 'y', '2', '3', '4', '5', '6', '7', '8']

# Set the RTSP stream source
source = "rtsp://admin:admin@192.168.1.88:554/substream"  # Replace with your RTSP URL

# Load YOLOv8 models for object and character detection

model_object = YOLO("weights/best.pt")
model_char = YOLO("weights/yolov8n_char_new.pt")

cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
cap.set(cv2.CAP_PROP_POS_FRAMES, 10)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open the RTSP stream.")
    exit()

# Define the output video properties (optional if saving images)
output_imagename = f'output_{timestamp}.jpg'

while cap.isOpened():
    success, img = cap.read()
    if success:
        tick = time.time()

        # Detect objects with YOLO model
        output = model_object(img, show=False, conf=0.7, stream=True)

        for result in output:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Convert to integers
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                confs = math.ceil((box.conf[0] * 100)) / 100
                cls_names = int(box.cls[0])

                if cls_names == 1:
                    cv2.putText(img, f'{confs}', (max(40, x2 + 5), max(40, y2 + 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 20, 255), thickness=1, lineType=cv2.LINE_AA)
                elif cls_names == 0:
                    cv2.putText(img, f'{confs}', (max(40, x1), max(40, y1)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(0, 20, 255), thickness=1, lineType=cv2.LINE_AA)

                # Check for plates and recognize characters
                if cls_names == 1:
                    char_display = []
                    plate_img = img[y1:y2, x1:x2]  # Crop plate from frame
                    plate_output = model_char(plate_img, conf=0.3)

                    bbox = plate_output[0].boxes.xyxy
                    cls = plate_output[0].boxes.cls

                    # Sort characters left to right
                    keys = cls.cpu().numpy().astype(int)
                    values = bbox[:, 0].cpu().numpy().astype(int)
                    dictionary = list(zip(keys, values))
                    sorted_list = sorted(dictionary, key=lambda x: x[1])

                    # Convert to string
                    for i in sorted_list:
                        char_class = i[0]
                        char_display.append(charclassnames[char_class])

                    char_result = 'Plate: ' + ''.join(char_display)

                    if len(char_display) == 8:
                        cv2.line(img, (max(40, x1 - 25), max(40, y1 - 10)), (x2 + 25, y1 - 10), (0, 0, 0), 20, lineType=cv2.LINE_AA)
                        cv2.putText(img, char_result, (max(40, x1 - 15), max(40, y1 - 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10, 50, 255), thickness=1, lineType=cv2.LINE_AA)

        tock = time.time()
        elapsed_time = tock - tick
        fps_text = "FPS: {:.2f}".format(1 / elapsed_time)
        text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        fps_text_loc = (img.shape[1] - text_size[0] - 10, text_size[1] + 10)
        cv2.putText(img, fps_text, fps_text_loc, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(10, 50, 255), thickness=2, lineType=cv2.LINE_AA)

        # Show detection
        cv2.imshow('detection', img)

        # Save the frame as an image
        cv2.imwrite('output/' + output_imagename, img)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print("Error: Could not get any frames from the RTSP stream.")
        break

cap.release()
cv2.destroyAllWindows()
