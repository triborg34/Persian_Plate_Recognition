import cv2
import os

# Function to crop the plate and save both the car image and the cropped plate
def crop_and_save_plate(img, box, plate_number, output_folder='output'):
    # Get bounding box coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

    # Crop the plate from the car image
    plate_img = img[y1:y2, x1:x2]

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the full car image
    car_img_path = os.path.join(output_folder, f'car_{plate_number}.jpg')
    cv2.imwrite(car_img_path, img)
    print(f"Car image saved at {car_img_path}")

    # Save the cropped plate image
    plate_img_path = os.path.join(output_folder, f'plate_{plate_number}.jpg')
    cv2.imwrite(plate_img_path, plate_img)
    print(f"Plate image saved at {plate_img_path}")

    # Print the plate number to the console
    print(f"Detected License Plate: {plate_number}")
