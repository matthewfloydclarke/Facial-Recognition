import cv2
import matplotlib.pyplot as plt

image_Path_Prefix = './Facial_Recognition/'
imagePath = 'image.jpg'
imagePath = image_Path_Prefix + imagePath

img = cv2.imread(imagePath)

# Prints height, width, and channels. 3 is outputted in channels as the image is color and RGB (red, green, blue) is 3 different channels used
print(img.shape)

# Convert image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

# Finds faces using "haarcascade_frontalface_default.xml", which is a frontal facing detection file
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# detectMultiScale is used to identify faces of different sizes in the input image
faces = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# Draws a border around the face and adds confidence text
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Mock confidence value for illustration
    confidence = 95  # Replace with actual confidence value if available
    confidence_text = f'Conf: {confidence}%'
    cv2.putText(img, confidence_text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Convert the image and border back to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(20, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
