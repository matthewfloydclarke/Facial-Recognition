# Referenced from: https://www.datacamp.com/tutorial/face-detection-python-opencv?dc_referrer=https%3A%2F%2Fwww.google.com%2F
import cv2

image_Path_Prefix = './Facial_Recognition/'
imagePath = 'image.jpg'
imagePath = image_Path_Prefix + imagePath

img = cv2.imread(imagePath)

# Prints height, width and channels. 3 is outputted in channels as the image is colour and RGB (red green blue) is 3 different channels used
print(img.shape)

# Convert iamge to greyscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray_image.shape)

# Finds faces using "haarcascade_frontalface_default.xml", which is a frontal facing detection file
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

)

# detectMultiScale is used to identify faces of different sizes in the input image
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# Draws a border round the face
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# Convert the image and border back to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Draw the iamge
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()