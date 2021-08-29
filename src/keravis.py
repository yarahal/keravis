import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils import find_closest_factors
from math import sqrt, log

def first_conv_filters(model,scale=5,title=None):
    '''
    Visualizes filters in the first convolutional layer in the model

    Parameters
    ----------
    model : keras Model

    Outputs
    -------
    A grid of the weights of the first convolutional layer
    '''
    # find the first convolutional layer
    first_conv_layer = None
    for layer in model.layers:
        if isinstance(layer,tf.keras.layers.Conv2D):
            first_conv_layer = layer
            break
    
    # get filters of convolutional layer
    filters = first_conv_layer.get_weights()[0]
    n_filters = filters.shape[3]
    
    # create grid and plot filters
    rows, cols = find_closest_factors(n_filters)
    n_pixels = filters.shape[0]
    fig, axs = plt.subplots(rows,cols,figsize=(scale*0.02*n_pixels*cols,scale*0.02*n_pixels*rows),gridspec_kw={'wspace':0.1, 'hspace':0.1})
    if title is not None:
        fig.suptitle(title)
    axs = axs.flatten()
    for i in range(n_filters):
        ax = axs[i]
        flter = filters[:,:,:,i]
        ax.imshow(flter,aspect='auto',interpolation='bilinear')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

def conv_activations(model,layer,test_img,title=None):
    '''
    Visualizes activations of a given convolutional layer for a given image

    Parameters
    ----------
    model : keras Model
    layer : str
        layer name
    test_img : ndarray
        image for which to look at activations of

    Outputs
    -------
    A grid of n_channels of activations of the given layer corresponding to the test image
    '''
    # get layer from model
    conv_layer = model.get_layer(layer)
    
    # create model whose output is output of conv_layer
    conv_model = tf.keras.Model(inputs=model.input,outputs=conv_layer.output)
    
    # retrieve activations of test_img
    activations = conv_model(np.expand_dims(test_img,0))
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

def maximally_activating_imgs(model,layer,n_neurons,X,k=5,channel=None,title=None):
    '''
    Visualizes top k images that maximize activations of random neurons in a given channel in a convolutional layer

    Parameters
    ----------
    model : keras Model
    layer : str
        layer name
    X : ndarray
        given set of images
    k : int
    channel : int
        channel for which to find maximally activating images

    Outputs
    -------
    An (n_neurons x k) grid of images that maximally activate randomly chosen neurons in a given channel in the given layer 
    '''
    #!TODO find patches
    #!TODO use dataiterator

    # get layer from model
    conv_layer = model.get_layer(layer)
    if channel is None:
        channel = np.random.randint(0,conv_layer.filters)
    
    # create model whose output is output of conv_layer
    conv_model = tf.keras.Model(inputs=model.input,outputs=conv_layer.output)
    
    # retrieve activations of X
    activations = conv_model(X)

    # choose random neurons
    random_neuron_idxs = zip(np.random.randint(0,activations.shape[1],n_neurons),np.random.randint(0,activations.shape[2],n_neurons))
    
    # get image indices that correspond to k-maximal activations of each neuron
    image_idxs = np.zeros((n_neurons,k))
    for i in range(n_neurons):
        image_idxs[i,:] = np.argsort(activations[:,random_neuron_idxs[0][i],random_neuron_idxs[1][i],channel])[:k]
    
    # create grid and plot images
    n_pixels = X.shape[1]
    fig, axs = plt.subplots(n_neurons,k,figsize=(0.02*n_pixels*k,0.02*n_pixels*n_neurons),gridspec_kw={'wspace':0.1, 'hspace':0.1})
    if title is not None:
        fig.suptitle(title)
    axs = axs.flatten()
    image_idxs = image_idxs.flatten()
    for i in range(k*n_neurons):
        ax = axs[i]
        ax.imshow(X[image_idxs[i],:,:,:],aspect='auto')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

def fc_knn(model,X,test_img,k,title=None):
    '''
    Visualizes k-images closest to a given test image based on activations of the last fully-connected layer

    Parameters
    ----------
    model : keras Model
    X : ndarray
        given set of images
    test_img : ndarray
        image for which to look for neighbors in X
    k : int
        k in KNN

    Outputs
    -------
    A horizontal grid of the test image alongside the k-nearest images
    '''
    #!TODO use dataiterator

    # find the fc layer
    fc_layer = model.layers[-3]
    
    # create model whose output is output of fc_layer
    fc_model = tf.keras.Model(inputs=model.input,outputs=fc_layer.output)

    # store fc layer for images in X
    fc_layer_activations = []
    for img in X:
        fc_layer_activations.append(np.array(fc_model(np.expand_dims(img,0))).flatten())
    fc_layer_activations = np.array(fc_layer_activations)
    
    # fit k-nearest neighbors model on X
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(fc_layer_activations,np.zeros(fc_layer_activations.shape[0]))
    
    # get k-nearest neighbors of test image
    test_img_activation = np.array(fc_model(np.expand_dims(test_img,0))).flatten()
    neighbors = knn.kneighbors(np.expand_dims(test_img_activation,0),return_distance=False).flatten()
    
    # plot result
    n_pixels = test_img.shape[0]
    fig, axs = plt.subplots(1,k+1,figsize=(0.02*n_pixels*(k+1),0.02*n_pixels),gridspec_kw={'wspace':0, 'hspace':0})
    if title is not None:
        fig.suptitle(title)
    axs = axs.flatten()
    ax = axs[0]
    ax.imshow(test_img,aspect='auto')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("test image")
    for i in range(k):
        ax = axs[i+1]
        neighbor_idx = neighbors[i]
        ax.imshow(X[neighbor_idx,:,:,:],aspect='auto')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

def fc_tsne(model,X,y,title=None):
    '''
    Visualizes activations of the last fully-connected layer on a set of images in 2-dimensional space using tSNE
        
    Parameters
    ---------
    model : keras Model
    X : ndarray
        given set of images
    y : ndarray
        given labels

    Outputs
    -------
    2-dimensional tSNE visualization of activations of last fully-connected layer corresponding to a given set of images X
    '''
    # find the fc layer
    fc_layer = model.layers[-2]
    
    # create model whose output is output of fc_layer
    fc_model = tf.keras.Model(inputs=model.input,outputs=fc_layer.output)

    # store fc layer for images in X
    fc_layer_activations = []
    for img in X:
        fc_layer_activations.append(np.array(fc_model(np.expand_dims(img,0))).flatten())
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

def fc_pca(model,X,y,title=None):
    '''
    Visualizes activations of the last fully-connected layer on a set of images in 2-dimensional space using PCA
    
    Parameters
    ---------
    model : keras Model
    X : ndarray
        given set of images
    y : ndarray
        given labels

    Outputs
    -------
    2-dimensional PCA visualization of activations of last fully-connected layer corresponding to a given set of images X
    '''
    # find the fc layer
    fc_layer = model.layers[-2]
    
    # create model whose output is output of fc_layer
    fc_model = tf.keras.Model(inputs=model.input,outputs=fc_layer.output)

    # store fc layer for images in X
    fc_layer_activations = []
    for img in X:
        fc_layer_activations.append(np.array(fc_model(np.expand_dims(img,0))).flatten())
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

def saliency_occlusion(model,test_img,class_idx,title=None):
    '''
    Visualizes the saliency map of an image using occulsion
    
    Parameters
    ---------
    model : keras Model
    test_img : ndarray
        image for which to find saliency map
    class_idx : int
        index of test_img label wrt to model output

    Outputs
    -------
    Saliency map of test_img by occlusion
    '''
    # get needed saliency map dimensions
    mask_width, mask_height = test_img[0]//4, test_img[1]//4
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

def saliency_backprop(model,test_img,title=None):
    '''
    Visualizes the saliency map of an image using backprop

    Parameters
    ----------
    model : keras Model
    test_img : ndarray
        image for which to find saliency map

    Outputs
    -------
    Saliency map of test_img by backprop
    '''

    # create layer to invert softmax activation
    invert_layer = tf.keras.layers.Lambda(lambda x : tf.math.log(x)+ log(10.))
    
    # create model whose output is output of invert_layer
    x = model(model.input)
    x = invert_layer(x)
    top_model = tf.keras.Model(inputs=model.input,outputs=x)
    
    # convert image to tensor
    tensor_img = tf.convert_to_tensor(np.expand_dims(test_img,0))
    
    # compute gradient of invert_layer wrt to input image 
    with tf.GradientTape() as tape:
        tape.watch(tensor_img)
        output = top_model(tensor_img)
    gradients = tape.gradient(output,tensor_img)
    saliency_map = np.max(np.abs(gradients[0,:,:,:]),axis=2)

    # plot result
    n_pixels = test_img.shape[0]
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(0.02*n_pixels*2,0.02*n_pixels),gridspec_kw={'wspace':0, 'hspace':0})
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