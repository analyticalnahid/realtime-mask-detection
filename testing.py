import time

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model


stream = cv2.VideoCapture(0)


yolo_model = load_model("model_data/yolo.h5")


def predict(sess, frame):

    # Preprocess your image
    image, image_data = preprocess_webcam_image(frame, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data,
                                                                                       K.learning_phase(): 0})
    # Print predictions info
    print('Found {} boxes'.format(len(out_boxes)))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    return np.array(image)


sess = K.get_session()

while True:
    # Capture frame-by-frame
    grabbed, frame = stream.read()
    if not grabbed:
        break

    # Run detection
    start = time.time()
    output_image = predict(sess, frame)
    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    # Display the resulting frame
    cv2.imshow('', output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
stream.release()
cv2.destroyAllWindows()