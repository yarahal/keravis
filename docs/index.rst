keravis API Documentation
=========================

Convolutional layer activations
*******************************

``conv_layer_activations(model, layer, test_img, nested_model=None, title=None)``
 Visualize activations of `layer` corresponding to `test_img` in a grid
*Args:*
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

2-dimensional feature space representations
*******************************************

``feature_space(model, dataset=None, X=None, y=None, kind='tsne', title=None)``
 Visualize feature space of `model` on a set of images `X` in 2-dimensional space using tSNE or PCA
*Args:*
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

Saliency maps
*************

``saliency_backprop(model, test_img, class_idx=0, title=None)``
 Visualize the saliency map of `test_img` using vanilla backprop
*Args:*
    model : keras.Model
        Model.
    test_img : ndarray
        Image for which to find saliency map of.
    class_idx : int, default=0
        Class index of image.
    title : str, default=None
        Title of the figure.

``saliency_guided_backprop(model, test_img, class_idx=0, title=None)``
 Visualize the saliency map of `test_img` using guided backprop
*Args:*
    model : keras.Model
        Model.
    test_img : ndarray
        Image for which to find saliency map of.
    class_idx : int, default=0
        Class index of image.
    title : str, default=None
        Title of the figure.

``saliency_occlusion(model, test_img, class_idx=0, title=None)``
 Visualize the saliency map of `test_img` using occlusion
*Args:*
    model : keras.Model
        Model.
    test_img : ndarray
        Image for which to find saliency map of.
    class_idx : int, default=0
        Class index of image.
    title : str, default=None
        Title of the figure.

Generated image that maximally activates classifier output
**********************************************************

``maximal_class_score_input(model, class_idx, dim, title=None)``
 Visualize a generated image corresponding to a maximal class score of `class_idx`
*Args:*
    model : keras.Model
        Model.
    class_idx : int
        Class index for which to find maximally activating image.
    dim : tuple
        (width,height,channels) of generated image.
    title : str, default=None
        Title of the figure.

Patches in a set of images that maximally activate an intermediate neuron
*************************************************************************

``maximally_activating_patches(model, layer, dataset=None, X=None, nested_model = None, channel=None, title=None)``
 Visualizes maximally activating patches in `X` of a random intermediate neuron in `layer`, `channel`
*Args:*
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