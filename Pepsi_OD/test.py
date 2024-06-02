import tensorflow as tf
import matplotlib.pyplot as plt
import os

model=tf.keras.models.load_model("D:\Python\ML\Projects\Pepsi_OD\pepsi.h5")
class_names=["Mountain Dew","Pepsi"]

def load_and_prep_image(filename, img_shape=512):
  """
  Reads an image from filename, turns it into a tensor and reshapes it
  to (img_shape, img_shape, colour_channels).
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode the read file into a tensor
  img = tf.image.decode_image(img)
  # Resize the image
  img = tf.image.resize(img, size=[515, img_shape])
  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

def pred_and_plot(model, filename, class_names=class_names):
    img = load_and_prep_image(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    if pred.max()<0.7:
        pred_class="Unknown"
    
    else:
        if len(pred[0]) > 1:
            pred_class = class_names[tf.argmax(pred[0])]
        else:
            pred_class = class_names[int(tf.round(pred[0]))]
    
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class} Prob:{pred.max()}")
    plt.axis(False)
    plt.show()

path="test_images"
files=[]
for r, d, f in os.walk(path):
   for file in f:
     files.append(os.path.join(r, file))
for f in files:
    pred_and_plot(model=model,filename=f)