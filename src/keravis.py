import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils import pixel_scaling, find_closest_factors, normalize, clone_function_1, clone_function_2
import cv2


def conv_layer_activations(model,
                           layer,
                           test_img,
                           nested_model=None,
                           title=None):
    '''
    Visualize activations of `layer` corresponding to `test_img` in a grid

    Parameters
    ----------
    model : keras.Model
        Model.
    layer : str
        Layer whose activations to visualize.
    test_img : ndarray
        Image for which to look at activations of.
    nested_model : str, default=None
        Name of nested model, if any.
    title : str, default=None
        Title of the figure.
    '''

    # create modified_model whose output is output of conv_layer
    if nested_model is None:
        conv_layer = model.get_layer(layer)
        modified_model = tf.keras.Model(
            inputs=model.input, outputs=conv_layer.output)
    else:
        nested_model = model.get_layer(nested_model)
        conv_layer = nested_model.get_layer(layer)
        x = model.input
        for layer in model.layers:
            if layer == nested_model:
                break
            x = layer(x)
        x = tf.keras.Model(nested_model.input, conv_layer.output)(x)
        modified_model = tf.keras.Model(inputs=model.input, outputs=x)

    # retrieve activations of test_img
    activations = modified_model(np.expand_dims(test_img, 0))
    n_channels = activations.shape[3]

    # create grid and plot activations
    rows, cols = find_closest_factors(n_channels)
    width, height = pixel_scaling(
        activations.shape[1])*cols, pixel_scaling(activations.shape[2])*rows
    fig, axs = plt.subplots(rows, cols, figsize=(
        width, height), gridspec_kw={'wspace': 0.1, 'hspace': 0.1})
    if title is not None:
        fig.suptitle(title)
    axs = axs.flatten()
    for i in range(n_channels):
        ax = axs[i]
        activation = activations[0, :, :, i]
        ax.imshow(activation, cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


def feature_space(model,
                  dataset=None,
                  X=None,
                  y=None,
                  kind='tsne',
                  title=None):
    '''
    Visualize feature space of `model` on a set of images `X` in 2-dimensional space using tSNE or PCA

    Parameters
    ----------
    model : keras.Model
        Model.
    dataset : keras.preprocessing.image.DataIterator, default=None
        Batched dataset.
        If given, X and y are ignored.
    X : ndarray, default=None
        Set of images.
    y : ndarray, default=None
        Set of labels.
    kind : str, default='tsne'
        Type of plot. One of 'tsne' or 'pca'.
    title : str, default=None
        Title of the figure.
    '''
    # find the fc layer
    fc_layer = model.layers[-2]

    # create model whose output is output of fc_layer
    modified_model = tf.keras.Model(
        inputs=model.input, outputs=fc_layer.output)

    # set X and y to a batch of images and labels if dataset is given
    if dataset is not None:
        X, y = dataset.next()

    # store fc_layer activations for images in X
    fc_layer_activations = []
    for img in X:
        fc_layer_activations.append(
            np.array(modified_model(np.expand_dims(img, 0))).flatten())
    fc_layer_activations = np.array(fc_layer_activations)

    # fit and transform TSNE on fc_layer_activations
    if kind == 'tsne':
        tsne = TSNE(n_components=2)
        transformed_X = tsne.fit_transform(fc_layer_activations)
    elif kind == 'pca':
        pca = PCA(n_components=2)
        transformed_X = pca.fit_transform(fc_layer_activations)

    # plot result
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.legend(*scatter.legend_elements())
    if title is not None:
        ax.set_title(title)


def saliency_backprop(model,
                      test_img,
                      class_idx=0,
                      title=None):
    '''
    Visualize the saliency map of `test_img` using vanilla backprop

    Parameters
    ----------
    model : keras.Model
        Model.
    test_img : ndarray
        Image for which to find saliency map of.
    class_idx : int, default=0
        Class index of image.
    title : str, default=None
        Title of the figure.
    '''

    # create modified_model with softmax activation of output layer removed to get raw class scores
    modified_model = tf.keras.models.clone_model(
        model, model.input, clone_function_1)
    modified_model.set_weights(model.get_weights())

    # convert image to tensor
    tensor_img = tf.convert_to_tensor(np.expand_dims(test_img, 0))

    # compute gradient of output wrt to input image
    with tf.GradientTape() as tape:
        tape.watch(tensor_img)
        output = modified_model(tensor_img)[:, class_idx]
    gradients = tape.gradient(output, tensor_img)
    saliency_map = np.max(np.abs(gradients[0, :, :, :]), axis=2)
    normalize(saliency_map)

    # plot result
    width, height = pixel_scaling(
        test_img.shape[0])*2, pixel_scaling(test_img.shape[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width,
                                                  height), gridspec_kw={'wspace': 0, 'hspace': 0})
    if title is not None:
        fig.suptitle(title)
    ax1.imshow(test_img, aspect='auto')
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("test image")
    ax2.imshow(saliency_map, aspect='auto',
               cmap='jet', interpolation='bilinear')
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("saliency map")


def saliency_guided_backprop(model,
                             test_img,
                             class_idx,
                             title=None):
    '''
    Visualize the saliency map of `test_img` using guided backprop

    Parameters
    ----------
    model : keras.Model
        Model.
    test_img : ndarray
        Image for which to find saliency map of.
    class_idx : int, default=0
        Class index of image.
    title : str, default=None
        Title of the figure.
    '''

    # create modified model for guided backprop
    modified_model = tf.keras.models.clone_model(
        model, model.input, clone_function_2)
    modified_model.set_weights(model.get_weights())

    # convert image to tensor
    tensor_img = tf.convert_to_tensor(np.expand_dims(test_img, 0))

    # compute gradient of output wrt to input image
    with tf.GradientTape() as tape:
        tape.watch(tensor_img)
        output = modified_model(tensor_img)[:, class_idx]
    gradients = tape.gradient(output, tensor_img)
    saliency_map = np.max(np.abs(gradients[0, :, :, :]), axis=2)
    normalize(saliency_map)

    # plot result
    width, height = pixel_scaling(
        test_img.shape[0])*2, pixel_scaling(test_img.shape[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width,
                                                  height), gridspec_kw={'wspace': 0, 'hspace': 0})
    if title is not None:
        fig.suptitle(title)
    ax1.imshow(test_img, aspect='auto')
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("test image")
    ax2.imshow(saliency_map, aspect='auto',
               cmap='jet', interpolation='bilinear')
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("saliency map")


def saliency_occlusion(model,
                       test_img,
                       class_idx,
                       title=None):
    '''
    Visualize the saliency map of `test_img` using occlusion

    Parameters
    ----------
    model : keras.Model
        Model.
    test_img : ndarray
        Image for which to find saliency map of.
    class_idx : int, default=0
        Class index of image.
    title : str, default=None
        Title of the figure.
    '''

    # find saliency map dimensions
    mask_width, mask_height = test_img.shape[0]//4, test_img.shape[1]//4
    mask_stride = 1
    width, height = (
        test_img.shape[0]-mask_width)//mask_stride, (test_img.shape[1]-mask_height)//mask_stride

    # create saliency map
    saliency_map = np.zeros((width, height))

    # fill saliency map by sliding mask across image
    for i in range(0, width, mask_stride):
        for j in range(0, height, mask_stride):
            img_clipped = np.copy(test_img)
            img_clipped[i:i+mask_width, j:j+mask_height, :] = 0
            saliency_map[i, j] = np.array(
                model(np.expand_dims(img_clipped, 0))).flatten()[class_idx]
    normalize(saliency_map)

    # plot result
    width, height = pixel_scaling(
        test_img.shape[0])*2, pixel_scaling(test_img.shape[1])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width,
                                                  height), gridspec_kw={'wspace': 0, 'hspace': 0})
    if title is not None:
        fig.suptitle(title)
    ax1.imshow(test_img, aspect='auto')
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("test image")
    ax2.imshow(saliency_map, aspect='auto',
               cmap='jet', interpolation='bilinear')
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("saliency map")


def maximal_class_score_input(model,
                              class_idx,
                              dim,
                              title=None):
    '''
    Visualize a generated image corresponding to a maximal class score of `class_idx`

    Parameters
    ----------
    model : keras.Model
        Model.
    class_idx : int
        Class index for which to find maximally activating image.
    dim : tuple
        (width,height,channels) of generated image.
    title : str, default=None
        Title of the figure.
    '''

    # create model with softmax activation of output layer removed
    modified_model = tf.keras.models.clone_model(
        model, model.input, clone_function_1)
    modified_model.set_weights(model.get_weights())

    # create random image
    img = np.random.randn(dim[0], dim[1], dim[2])

    # convert image to tensor
    tensor_img = tf.convert_to_tensor(np.expand_dims(img, 0), dtype='float32')

    # maximize class score wrt image pixels
    for n in tf.range(2048):
        with tf.GradientTape() as tape:
            tape.watch(tensor_img)
            output = modified_model(tensor_img)[0, class_idx]
        gradient = tape.gradient(output, tensor_img)
        gradient /= tf.math.reduce_std(gradient) + 1e-7
        tensor_img += 0.05 * gradient
        # regularize
        if n % 100 == 0:
            tensor_img = tfa.image.gaussian_filter2d(tensor_img, sigma=1)

    result = tensor_img[0].numpy()
    normalize(result)

    # plot result
    fig, ax = plt.subplots(
        figsize=(pixel_scaling(dim[0]), pixel_scaling(dim[1])))
    if title is not None:
        ax.set_title(title)
    ax.imshow(result, aspect='auto')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def maximally_activating_patches(model,
                                 layer,
                                 dataset=None,
                                 X=None,
                                 nested_model=None,
                                 channel=None,
                                 title=None):
    '''
    Visualizes maximally activating patches in `X` of a random intermediate neuron in `layer`, `channel`

    Parameters
    ----------
    model : keras.Model
        Model.
    layer : str
        Layer whose activations to visualize.
    dataset : keras.preprocessing.image.DataIterator, default=None
        Batched dataset.
        If given, X and y are ignored.
    X : ndarray, default=None
        Set of images.
    nested_model : str, default=None
        Name of nested model, if any.
    channel : int, default=None
        Channel index. 
        If not given, channel is randomly sampled.
    title : str, default=None
        Title of the figure.
    '''

    # create modified_model whose output is output of conv_layer
    if nested_model is None:
        conv_layer = model.get_layer(layer)
        modified_model = tf.keras.Model(
            inputs=model.input, outputs=conv_layer.output)
    else:
        nested_model = model.get_layer(nested_model)
        conv_layer = nested_model.get_layer(layer)
        x = model.input
        for layer in model.layers:
            if layer == nested_model:
                break
            x = layer(x)
        x = tf.keras.Model(nested_model.input, conv_layer.output)(x)
        modified_model = tf.keras.Model(inputs=model.input, outputs=x)

    # create modified model for guided backprop
    modified_model = tf.keras.models.clone_model(
        modified_model, modified_model.input, clone_function_2)
    modified_model.set_weights(modified_model.get_weights())

    # pick a random channel if channel is not given
    if channel is None:
        channel = np.random.randint(0, conv_layer.filters)
    # pick a random neuron
    random_neuron = (np.random.randint(
        0, conv_layer.output_shape[1]), np.random.randint(0, conv_layer.output_shape[2]))

    # set X to a batch of images if dataset is given
    if dataset is not None:
        X, _ = dataset.next()

    # find and sort neuron activations of images
    neuron_activations = []
    for img in X:
        neuron_activations.append((np.array(modified_model(np.expand_dims(img, 0)))[
                                  0, random_neuron[0], random_neuron[1], channel]))
    image_idxs = np.argsort(neuron_activations)

    width, height = pixel_scaling(X.shape[1])*5, pixel_scaling(X.shape[2])
    fig, axs = plt.subplots(1, 5, figsize=(width, height), gridspec_kw={
                            'wspace': 0, 'hspace': 0})
    if title is not None:
        fig.suptitle(title)
    axs = axs.flatten()
    # get 5 images with highest activation values
    for k in range(len(image_idxs)-1, len(image_idxs)-6, -1):
        ax = axs[k-len(image_idxs)+1]
        img = X[image_idxs[k]]
        # convert image to tensor
        tensor_img = tf.convert_to_tensor(np.expand_dims(img, 0))
        # compute gradient of neuron wrt image
        with tf.GradientTape() as tape:
            tape.watch(tensor_img)
            output = modified_model(tensor_img)[
                0, random_neuron[0], random_neuron[1], channel]
        gradient = tape.gradient(output, tensor_img)
        gradient = gradient[0].numpy()
        gradient[gradient < 1e-5] = 0
        # find receptive field of neuron in image
        if gradient.shape[2] == 3:
            gradient = np.mean(gradient, axis=2)
        bounding_rect_coord = cv2.boundingRect(cv2.findNonZero(gradient))
        bounding_rect = ptc.Rectangle((bounding_rect_coord[0], bounding_rect_coord[1]), bounding_rect_coord[2],
                                      bounding_rect_coord[3], linewidth=2, edgecolor='k', facecolor='none')
        # plot image and patch
        ax.imshow(img, aspect='auto')
        ax.add_patch(bounding_rect)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
