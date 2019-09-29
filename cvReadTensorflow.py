import cv2

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('C:/Users/dylee/Archive/unet/log/frozen_model.pb')

# Input image
# img = cv2.imread('nails/src/00_001.png')
img = cv2.imread('C:/Users/dylee/Pictures/handDataset/picture/20190916_142844.jpg')


rows, cols, channels = img.shape
# img = img/255
# Use the given image as input, which needs to be blob(s).
input = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(224, 224), swapRB=True, crop=False)
tensorflowNet.setInput(input)

# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()

# Loop on the outputs
# for detection in networkOutput[0, 0]:
#
#     score = float(detection[2])
#     if score > 0.2:
#         left = detection[3] * cols
#         top = detection[4] * rows
#         right = detection[5] * cols
#         bottom = detection[6] * rows
#
#         # draw a red rectangle around detected objects
#         cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

# Show the image with a rectagle surrounding the detected objects
cv2.imshow('Image', cv2.resize(img,(224,224)))
cv2.imshow('Segmentation', networkOutput[0,0])
cv2.waitKey()
cv2.destroyAllWindows()