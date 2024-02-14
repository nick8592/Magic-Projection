# [Assignment - Magic Projection](https://jackybaltes.github.io/computer-vision/assignment_projection_magic)

## Trick 1: Dis/appearing Coins

> The simplest trick is to wave your hands over the field and make coins magically appear and/or disappear. Implement a program that tracks the hands of the magician and devise some way in which the user can add or remove coins. For example, by spreading their hands to signal a change.

Use `Torchvision` pre-trained model for coin detection, and use `Mediapipe` library for hand detection. The orientation of the thumb is defined by the difference between the thumb position and the wrist position in the viewport.

If there has coin and hand is detected,

- thumb point to the `right`, coins `disappear`
- thumb point to the `left`, coins `appear`

### Quick Start

```
python coin.py
```

### References

[Faster R-CNN model with a ResNet-50-FPN](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)  
[Hand landmarks detection guide](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

## Trick 2: Changing Cards

> A more advanced trick tracks white playing cards and replaces the view of the white card with a color change or by replacing the card. In this case, your program must track the card and then replace the white card with the image of a card that is suitably adapted to match the environment.

Use the `HoughLines` function of the `OpenCV` library to detect seperating line. Convert the image to HSV color space and customize the upper/lower bounds of the region to filter out the specified color. In this piece, if the object is to the left of the seperating line, the color will change from blue to red; otherwise, the color will not change.

### Quick Start

```
python card.py
```

### References

[Hough Line Transform](https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html)  
[Changing Colorspaces](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)

## Trick 3: Tracking Dice (Incomplete)

> In this trick, the user predicts the number that will appear on a single or multiple dice. Another trick is for the user to throw a dice. In this case, the system must project three sides viewable from a dice.

Use `D6 Dice - Images` dataset which has many dice images. The code can track the position of each dices from a top-down view images, the bounding of each dice are mark with green line and corners mark with red dot. A function can print out all combinations for a target_sum using numbers of dices.

- [x] Detect dices position
- [x] Get dices corner's coordinate
- [x] Generate target_sum combinations
- [ ] Project new number on the dices
- [ ] Real-Time capability

### Quick Start

```
python dice.py <target_sum>

(e.g.)
python dice.py 10
```

### References

[D6 Dice - Images](https://www.kaggle.com/datasets/koryakinp/d6-dices-images)
