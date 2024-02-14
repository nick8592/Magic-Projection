import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import mediapipe as mp

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Create a function to perform inference and draw bounding boxes for label number 85
def detect_coins(frame, model):
    # Convert the frame to a PyTorch tensor
    frame = F.to_tensor(frame).unsqueeze(0)

    with torch.no_grad():
        prediction = model(frame)

    # Get bounding box coordinates and labels
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    # Filter boxes for label number 85
    coin_boxes = [box for box, label in zip(boxes, labels) if label == 85]

    return coin_boxes

# hands variable
STATIC_IMAGE_MODE = False
MAX_NUM_HANDS = 1
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3
# choose mediapipe solution
mpHands = mp.solutions.hands
# set hands variable
hands = mpHands.Hands(STATIC_IMAGE_MODE,
                      MAX_NUM_HANDS,
                      MODEL_COMPLEXITY,
                      MIN_DETECTION_CONFIDENCE,
                      MIN_TRACKING_CONFIDENCE)
mpDraw = mp.solutions.drawing_utils
# draw circle on articulation point (關節點)
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
# draw line between articulation point
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with label number 85 in the current frame
    coin_boxes = detect_coins(frame, model)

    # Draw bounding boxes for label number 85
    offset = 200
    scale = 10
    # for box in coin_boxes:
    #     x1, y1, x2, y2 = map(int, box)
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.rectangle(frame, (x1 + offset, y1), (x2 + offset, y2), (0, 0, 255), 2)

    # Hand detection
    result = hands.process(frame)
    imgHeight = frame.shape[0]
    imgWidth = frame.shape[1]
    if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                # mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                for i, lm in enumerate(handLms.landmark):
                    # convert (x, y) percentage to real (x, y) coordinate in window
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
 
                    # wrist
                    if i == 0:
                        xWrist = xPos
                        # cv2.circle(frame, (xPos, yPos), 20, (255, 0, 0), cv2.FILLED)
                    
                    # thumb
                    if i == 4:
                        xThumb = xPos
                        # cv2.circle(frame, (xPos, yPos), 20, (255, 0, 0), cv2.FILLED)
                    
                    if i > 4:
                        if xWrist > xThumb:
                            # thumb point to the right, coins disappear
                            print("disappear")
                            for box in coin_boxes:
                                # Cover the region (x1, y1, x2, y2) with the offset region
                                x1, y1, x2, y2 = map(int, box)
                                frame[int(y1-scale):int(y2+scale), int(x1-scale):int(x2+scale)] = frame[int(y1-scale):int(y2+scale), int((x1+offset)-scale):int((x2+offset)+scale)]
                        else:
                            print("show")

                    
    # Display the frame with bounding boxes
    cv2.imshow("Coin Trick", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
