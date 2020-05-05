import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
import cv2
import numpy as np

def DrawEvolution(history):
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    for hist in history:
        acc += hist.history['accuracy']
        val_acc += hist.history['val_accuracy']
        loss += hist.history['loss']
        val_loss += hist.history['val_loss']
    print('Test: Accuracy mean/std - {}/{}'.format(np.mean(np.array(val_acc)),np.std(np.array(val_acc))))
    print('Test: Loss mean/std - {}/{}'.format(np.mean(np.array(val_loss)),np.std(np.array(val_loss))))
    print('Train: Accuracy mean/std - {}/{}'.format(np.mean(np.array(acc)),np.std(np.array(acc))))
    print('Train: Loss mean/std - {}/{}'.format(np.mean(np.array(loss)),np.std(np.array(loss))))
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def VisualizeIntermedianActivation(layer_cnt, model, img):
    layer_outputs = [layer.output for layer in model.layers[:layer_cnt]]
    activation_model = models.Model(inputs = model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)

    layer_names = []
    for layer in model.layers[:layer_cnt]:
        layer_names.append(layer.name)
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,:,:,col*images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, row * size : (row + 1)*size] = channel_image
        scale = 1./size
        plt.figure(figsize = (scale*display_grid.shape[1], scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap = 'viridis')

@tf.function
def CAM(model, layer_name, img, activator = 0):
    layer_output = model.get_layer(layer_name).output
    activation_model = models.Model(inputs = model.input, outputs = [model.output, layer_output])
    with tf.GradientTape() as tape:
        y, x = activation_model(img,training = False)
        if activator == 1:
            y = 1.0 - y
    z = tape.gradient(y, x)
    z = tf.reduce_mean(z, axis = (0,1,2))
    return tf.nn.relu(tf.reduce_mean(z * x, axis = (0,3)))

def GetActivatedRegion(model, layer_name, img, activator = 0):
    heatmap = CAM(model, layer_name, img, activator).numpy()
    heatmap = np.maximum(heatmap, 0.0)
    heatmap /= np.max(heatmap)
    img = np.reshape(img, (224,224,3))
    img = np.uint8(img * 255)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.4 + img)
    return superimposed_img
