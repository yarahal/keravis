import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils import find_closest_factors
from math import sqrt, log

def conv_layer_activations(model,
                           layer,
                           test_img,
                           title=None):
    '''
    Visualizes activations of a given convolutional layer for a given image

    Parameters
    ----------
    model : keras Model
    layer : keras Layer
        Layer
    test_img : ndarray
        Image for which to look at activations of

    Outputs
    -------
    A grid of activations of the given layer corresponding to the test image
    '''

    # create model whose output is output of conv_layer
    modified_model = tf.keras.Model(inputs=model.input,outputs=layer.output)
    
    # retrieve activations of test_img
    activations = modified_model(np.expand_dims(test_img,0))
    n_channels = activations.shape[3]
    
    # create grid and plot filters
    rows, cols = find_closest_factors(n_channels)
    n_pixels = activations.shape[1]
    fig, axs = plt.subplots(rows,cols,figsize=(0.02*n_pixels*cols,0.02*n_pixels*rows),gridspec_kw={'wspace':0.1, 'hspace':0.1})
    if title is not None:
        fig.suptitle(title)
    axs = axs.flatten()
    for i in range(n_channels):
        ax = axs[i]
        activation = activations[0,:,:,i]
        ax.imshow(activation,cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([]) 

def feature_space_tsne(model,
                       dataset=None,
                       X=None,
                       y=None,
                       title=None):
    '''
    Visualizes activations of the last fully-connected layer on a set of images in 2-dimensional space using tSNE
        
    Parameters
    ----------
    model : keras Model
    dataset : keras DataIterator
        Batched dataset
        If given, X and y are ignored.
    X : ndarray
        Set of images
    y : ndarray
        Set of labels

    Outputs
    -------
    2-dimensional tSNE visualization of activations of last fully-connected layer before the classifier corresponding to a batch in dataset or to a given set of images X.
    '''
    # find the fc layer
    fc_layer = model.layers[-2]
    
    # create model whose output is output of fc_layer
    modified_model = tf.keras.Model(inputs=model.input,outputs=fc_layer.output)

    # set X and y to a batch of images and labels if dataset is given
    if dataset is not None:
        X, y = dataset.next()

    # store fc layer for images in X
    fc_layer_activations = []
    for img in X:
        fc_layer_activations.append(np.array(modified_model(np.expand_dims(img,0))).flatten())
    fc_layer_activations = np.array(fc_layer_activations)
    
    # fit and transform TSNE on X
    tsne = TSNE(n_components=2)
    transformed_X = tsne.fit_transform(fc_layer_activations)
    
    # plot result
    fig, ax = plt.subplots(figsize=(7,7))
    scatter = ax.scatter(transformed_X[:,0],transformed_X[:,1],c=y)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.legend(*scatter.legend_elements())
    if title is not None:
        ax.set_title(title)

def feature_space_pca(model,
                      dataset=None,
                      X=None,
                      y=None,
                      title=None):
    '''
    Visualizes activations of the last fully-connected layer on a set of images in 2-dimensional space using PCA
    
    Parameters
    ----------
    model : keras Model
    dataset : keras DataIterator
        Batched dataset
        If given, X and y are ignored.
    X : ndarray
        Set of images
    y : ndarray
        Set of labels

    Outputs
    -------
    2-dimensional PCA visualization of activations of last fully-connected layer before the classifier corresponding to a batch in dataset or to a given set of images X.
    '''
    # find the fc layer
    fc_layer = model.layers[-2]
    
    # create model whose output is output of fc_layer
    modified_model = tf.keras.Model(inputs=model.input,outputs=fc_layer.output)

    # set X and y to a batch of images and labels if dataset is given
    if dataset is not None:
        X, y = dataset.next()

    # store fc layer for images in X
    fc_layer_activations = []
    for img in X:
        fc_layer_activations.append(np.array(modified_model(np.expand_dims(img,0))).flatten())
    fc_layer_activations = np.array(fc_layer_activations)
    
    # fit and transform PCA on X
    pca = PCA(n_components=2)
    transformed_X = pca.fit_transform(fc_layer_activations)
    
    # plot result
    fig, ax = plt.subplots(figsize=(7,7))
    scatter = ax.scatter(transformed_X[:,0],transformed_X[:,1],c=y)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.legend(*scatter.legend_elements())
    if title is not None:
        ax.set_title(title)

def saliency_occlusion(model,
                       test_img,
                       class_idx,
                       title=None):
    '''
    Visualizes the saliency map of an image using occulsion
    
    Parameters
    ----------
    model : keras Model
    test_img : ndarray
        Image for which to find saliency map
    class_idx : int
        Index of test_img label wrt to model output

    Outputs
    -------
    Saliency map of test_img by occlusion
    '''
    # get needed saliency map dimensions
    mask_width, mask_height = test_img.shape[0]//4, test_img.shape[1]//4
    mask_stride = 1
    width, height = (test_img.shape[0]-mask_width)//mask_stride, (test_img.shape[1]-mask_height)//mask_stride
    
    # create saliency map
    saliency_map = np.zeros((width,height))
    
    # fill saliency map by moving mask across image
    for i in range(0,width,mask_stride):
        for j in range(0,height,mask_stride):
            img_clipped = np.copy(test_img)
            img_clipped[i:i+mask_width,j:j+mask_height,:] = 0
            saliency_map[i,j] = np.array(model(np.expand_dims(img_clipped,0))).flatten()[class_idx]
    
    # plot result
    n_pixels_img = test_img.shape[0]
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(0.02*n_pixels_img*2,0.02*n_pixels_img),gridspec_kw={'wspace':0, 'hspace':0})
    if title is not None:
        fig.suptitle(title)
    ax1.imshow(test_img,aspect='auto')
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("test image")
    ax2.imshow(saliency_map,aspect='auto',cmap='jet',interpolation='bilinear')
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("saliency map")

def saliency_backprop(model,
                      test_img,
                      title=None):
    '''
    Visualizes the saliency map of an image using backprop

    Parameters
    ----------
    model : keras Model
    test_img : ndarray
        Image for which to find saliency map

    Outputs
    -------
    Saliency map of test_img by backprop
    '''

    # create layer to invert softmax activation
    invert_layer = tf.keras.layers.Lambda(lambda x : tf.math.log(x)+ log(10.))
    
    # create model whose output is output of invert_layer
    x = model(model.input)
    x = invert_layer(x)
    modified_model = tf.keras.Model(inputs=model.input,outputs=x)
    
    # convert image to tensor
    tensor_img = tf.convert_to_tensor(np.expand_dims(test_img,0))
    
    # compute gradient of invert_layer wrt to input image 
    with tf.GradientTape() as tape:
        tape.watch(tensor_img)
        output = modified_model(tensor_img)
    gradients = tape.gradient(output,tensor_img)
    saliency_map = np.max(np.abs(gradients[0,:,:,:]),axis=2)

    # plot result
    n_pixels = test_img.shape[0]
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(0.02*n_pixels*2,0.02*n_pixels),gridspec_kw={'wspace':0, 'hspace':0})
    if title is not None:
        fig.suptitle(title)
    ax1.imshow(test_img,aspect='auto')
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("test image")
    ax2.imshow(saliency_map,aspect='auto',cmap='jet',interpolation='bilinear')
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("saliency map")

def conv_features_backprop(model,
                          layer,
                          test_img,
                          channel=None,
                          n_neurons=10,
                          title=None):
    '''
    Visualizes gradients of intermediate neurons in the given layer corresponding to a test image using backprop

    Parameters
    ----------
    model : keras Model
    layer : keras Layer
        Layer 
    test_img : ndarray
        Image for which to look at intermediate features
    channel : int
    n_neurons : int

    Outputs
    -------
    A grid of gradients of n_neurons random neurons in layer wrt to test_img
    '''
    
    # create model whose output is output of layer
    modified_model = tf.keras.Model(inputs=model.input,outputs=layer.output)
    
    # convert image to tensor
    tensor_img = tf.convert_to_tensor(np.expand_dims(test_img,0))

    # choose a random channel if not given and random neurons
    if channel is None:
        channel = np.random.randint(0,layer.filters)
    random_neuron_idxs = zip(np.random.randint(0,layer.output_shape[1],n_neurons),np.random.randint(0,layer.output_shape[2],n_neurons))
    
    # record forward prop of image 
    with tf.GradientTape() as tape:
        tape.watch(tensor_img)
        output = modified_model(tensor_img)

    # compute and plot gradients of neurons wrt pixels of input image
    rows, cols = find_closest_factors(n_neurons)
    n_pixels = test_img.shape[0]
    fig, axs = plt.subplots(1,2,figsize=(0.02*n_pixels*cols,0.02*n_pixels*rows),gridspec_kw={'wspace':0, 'hspace':0})
    if title is not None:
        fig.suptitle(title)
    axs = axs.flatten()
    image_idxs = image_idxs.flatten()
    for k in range(n_neurons):
        ax = axs[k]
        i, j = random_neuron_idxs[0][k], random_neuron_idxs[1][k]
        feature_gradients = tape.gradient(output[:,i,j,channel],tensor_img)
        feature_gradients = np.max(np.abs(feature_gradients[0,:,:,:]),axis=2)
        ax.imshow(feature_gradients,aspect='auto')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])