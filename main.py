from ultralytics import YOLO
import cv2
import math
import time
import datetime

# Get the current timestamp for output names
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Define the file name with the timestamp
file_name = f'output_{timestamp}.jpg'
classnames = ['car', 'plate']
charclassnames = ['0','9','b','d','ein','ein','g','gh','h','n','s','1','malul','n','s','sad','t','ta','v','y','2'
                  ,'3','4','5','6','7','8']

# Connect to your IP camera using the RTSP stream
source = "rtsp://username:password@camera_ip_address:554/stream"  # Replace this with your IP camera's RTSP URL

# Load YOLOv8 model
model_object = YOLO("weights/best.pt")
model_char = YOLO("weights/yolov8n_char_new.pt")

# Open the RTSP stream
cap = cv2.VideoCapture(source)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open the RTSP stream.")
    exit()

# Define the output video properties
output_videoname = f'output_{timestamp}.mp4'
output_imagename = f'output_{timestamp}.jpg'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames > 1:
    video_writer = cv2.VideoWriter('output/' + output_videoname, fourcc, fps, (frame_width, frame_height))

    # Do inference on video from the RTSP stream
    while cap.isOpened():
        success, img = cap.read()
        if success:
            # Detect objects with yolov8 model
            tick = time.time()
            output = model_object(img, show=False, conf=0.7, stream=True)

            # Extract bounding box and class names
            for i in output:
                bbox = i.boxes
                for box in bbox:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    confs = math.ceil((box.conf[0] * 100)) / 100
                    cls_names = int(box.cls[0])

                    # Add class-specific text (confidence values)
                    if cls_names == 1:
                        cv2.putText(img, f'{confs}', (max(40, x2 + 5), max(40, y2 + 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 20, 255), thickness=1, lineType=cv2.LINE_AA)
                    elif cls_names == 0:
                        cv2.putText(img, f'{confs}', (max(40, x1), max(40, y1)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(0, 20, 255), thickness=1, lineType=cv2.LINE_AA)

                    # Check plate to recognize characters
                    if cls_names == 1:
                        char_display = []
                        plate_img = img[y1:y2, x1:x2]  # Crop plate from frame
                        plate_output = model_char(plate_img, conf=0.3)  # Detect characters of plate
                        
                        # Extract bounding box and class names for characters
                        bbox = plate_output[0].boxes.xyxy
                        cls = plate_output[0].boxes.cls

                        # Sort characters from left to right
                        keys = cls.cpu().numpy().astype(int)
                        values = bbox[:, 0].cpu().numpy().astype(int)
                        dictionary = list(zip(keys, values))
                        sorted_list = sorted(dictionary, key=lambda x: x[1])

                        for i in sorted_list:
                            char_class = i[0]
                            char_display.append(charclassnames[char_class])

                        char_result = 'Plate: ' + ''.join(char_display)

                        # Display detected plate characters
                        if len(char_display) == 8:
                            cv2.line(img, (max(40, x1 - 25), max(40, y1 - 10)), (x2 + 25, y1 - 10), (0, 0, 0), 20, lineType=cv2.LINE_AA)
                            cv2.putText(img, char_result, (max(40, x1 - 15), max(40, y1 - 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10, 50, 255), thickness=1, lineType=cv2.LINE_AA)

            # Display frame and FPS
            tock = time.time()
            elapsed_time = tock - tick
            fps_text = "FPS: {:.2f}".format(1 / elapsed_time)
            text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            fps_text_loc = (frame_width - text_size[0] - 10, text_size[1] + 10)
            cv2.putText(img, fps_text, fps_text_loc, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(10, 50, 255), thickness=2, lineType=cv2.LINE_AA)

            cv2.imshow('detection', img)
            video_writer.write(img)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

else:
    print("Error: Could not get any frames from the RTSP stream.")
