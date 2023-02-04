import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
import random

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x.reshape(-1,28,28,1).astype(np.float32)/255.
test_x = test_x.reshape(-1, 28, 28, 1).astype(np.float32) / 255.

bg_source = np.random.rand(1000,1000,1).astype(np.float32)
bg_source = filters.gaussian(bg_source, 4)
bg_source = (bg_source - np.min(bg_source))/(np.max(bg_source)-np.min(bg_source))
bg_source = np.clip(bg_source, 0,1)

plt.imshow(bg_source[...,0], 'gray')
plt.show()

def convert_to_loc_ds(x, y, bg_source, new_size):
    x_new = np.zeros((x.shape[0], new_size[0], new_size[1], 1), dtype=np.float32)
    y_new = np.zeros((y.shape[0], 10, 5), dtype=np.float32)
    y_new[:, :, 1:] = -1

    rh = float(x.shape[1]) / new_size[0]
    rw = float(x.shape[2]) / new_size[1]

    for i in range(x.shape[0]):
        oh = random.randint(0, bg_source.shape[0] - new_size[0])
        ow = random.randint(0, bg_source.shape[1] - new_size[1])
        x_new[i] = bg_source[oh:oh + new_size[0], ow:ow + new_size[1], :]
        oh = random.randint(0, x_new[i, ...].shape[0] - x[i].shape[0])
        ow = random.randint(0, x_new[i, ...].shape[1] - x[i].shape[1])
        x_new[i, oh:oh + x[i].shape[0], ow:ow + x[i].shape[1], :] += x[i]
        x_new[i] = np.clip(x_new[i], 0, 1)
        ry = float(oh) / new_size[0]
        rx = float(ow) / new_size[1]
        y_new[i][y[i]] = [1.0, ry, rx, rh, rw]

    return x_new, y_new

new_size = (64, 64)

train_x_new, train_y_new = convert_to_loc_ds(train_x, train_y, bg_source, new_size)
test_x_new, test_y_new = convert_to_loc_ds(test_x, test_y, bg_source, new_size)

def show_prediction(x, logits):
    import matplotlib.patches as patches

    pred_cls = np.argmax(logits[:,0])
    ry, rx, rh, rw = logits[pred_cls][1:]

    box_y = int(ry * x.shape[0])
    box_x = int(rx * x.shape[1])
    box_w = int(rw * x.shape[0])
    box_h = int(rh * x.shape[0])

    fig,ax = plt.subplots(1)
    ax.imshow(x[...,0], 'gray', vmin=0, vmax=1,)
    rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()
    print('Prediction: {}'.format(pred_cls))
    print('Box: {}'.format((box_x, box_y, box_w, box_h)))

i = random.randint(0, train_x_new.shape[0])
show_prediction(train_x_new[i], train_y_new[i])

class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, (5, 5), activation=tf.nn.relu, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(128, (5, 5), activation=tf.nn.relu, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(128, (5, 5), activation=tf.nn.relu, padding='same')
        self.fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(5 * 10, activation=None)
        self.max_pool = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')

    def call(self, inp):
        out = self.conv1(inp)
        out = self.max_pool(out)
        out = self.conv2(out)
        out = self.max_pool(out)
        out = self.conv3(out)
        out = self.max_pool(out)
        out = self.conv4(out)
        out = self.max_pool(out)
        out = tf.keras.layers.Flatten()(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = tf.reshape(out, (-1, out.shape[1] // 5, 5))

        return out

model = Model()

def loss_cls(labels, logits):
    logits_cls = logits[:,:,0]
    labels_cls = labels[:,:,0]
    return tf.nn.softmax_cross_entropy_with_logits(labels_cls, logits_cls)

def loss_box(labels, logits):
    logits_box = logits[:,:,1:]
    labels_box = labels[:,:,1:]
    return tf.keras.losses.MeanSquaredError()(labels_box, logits_box)

LAMBDA = 10.0
def loss_composit(labels, logits):
    return loss_cls(labels, logits) + LAMBDA * loss_box(labels, logits)

NUM_EPOCHS = 20
BATCH_SIZE = 64
# change LAMBDA to 10.0

model.compile(optimizer='adam', loss=loss_composit)

hist = model.fit(train_x_new, train_y_new, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

plt.plot(hist.history['loss'])

i = random.randint(0, test_x_new.shape[0])
sample = test_x_new[i]
pred = model.predict(sample[None, ...])[0, ...]
show_prediction(sample, pred)