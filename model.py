import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import load_model
import cv2

'''mnist = keras.datasets.mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(64, activation='relu'),  # hidden layer (3)
    keras.layers.Dense(10, activation='softmax') # output layer (4)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

if os.path.isfile('handwritten_digits.h5') is False:
    model.save('handwritten_digits.h5')'''

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

model = load_model('handwritten_digits.h5')

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

drawing = False  # true if mouse is pressed
ix, iy = 5, 5


def process(img):
    img = img / 255.0
    img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    img = img.reshape(1, 28, 28)
    return img


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 20, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 20, (255, 255, 255), -1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.rectangle(img, (0, 0), (512, 512), (0, 0, 0), -1)


img = np.zeros((512, 512, 1), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):  # break out of loop
        print("Code Ended")
        break
    elif k == ord('p'):  # predict the image
        # convert image to img that can be understood by the model
        newimg = process(img)
        prediction = model.predict(newimg)
        print("Model Predicts this number as : {} with accuracy of {}%".format(classes[np.argmax(prediction)],
                                                                               np.amax(prediction) * 100))
cv2.destroyAllWindows()
