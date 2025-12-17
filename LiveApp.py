import cv2
from PIL import Image
from SVM_inference import classify_image
 
def main():
    camera = cv2.VideoCapture(0)
    frame_counter = 0
    skip_frames = 30
    while True:
        
        success, frame = camera.read()
        if not success:
            break
        if frame_counter % skip_frames == 0:
            pil_image = frame_to_image(frame)


        class_name, confidence = classify_image(pil_image)

        
        # show result
        cv2.putText(frame, f"{class_name}: {confidence:.2f}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_counter += 1
        
        # display
        cv2.imshow('Material Classifier', frame)
        
        # quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def frame_to_image(frame):
    # 1. Convert BGR to RGB (OpenCV → PIL color difference)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 2. Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    return pil_image



if __name__ == "__main__":
    main()