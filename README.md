# keravis

keravis is a high-level API for ConvNet visualizations in Keras. As of v1.0, it supports visualizations of

1. Convolutional layer activations
2. 2-dimensional feature space representations
3. Saliency maps (vanilla backprop, guided backprop, and occlusion)
4. Synthetic maximally-activating images of classifier output
5. Maximally activating patches of an intermediate neuron in a set of images

with support for nested pretrained models.

This is a hobby project that was inspired by lecture 14 of Stanford's CS231n: Convolutional Neural Networks for Visual Recognition http://cs231n.stanford.edu/. It is not yet optimized for serious use (see keras-vis instead).

## Installation
## Usage

## MNIST Examples
```python
from keravis import feature_space
feature_space(model,X=x_test[:5000],y=y_test[:5000],kind='tsne')
```
![MNIST_TSNE](https://user-images.githubusercontent.com/65565946/132919099-7468290d-bc5d-4cfe-9cd4-22bea87f3849.png)

```python
from keravis import saliency_backprop
saliency_backprop(model,test_img,class_idx=7)
```
![saliency_1](https://user-images.githubusercontent.com/65565946/132919163-b4c4e5a8-a410-451c-9f23-7efbc3076110.png)

```python
from keravis import saliency_guided_backprop
saliency_guided_backprop(model,test_img,class_idx=7)
```
![saliency](https://user-images.githubusercontent.com/65565946/132919195-76ede1b1-a410-418e-ab75-1d2fa8e355bd.png)

```python
from keravis import classifier_gradient_ascent
classifier_gradient_ascent(model,class_idx=5,dim=(28,28,1))
```
![gradient_ascent_5](https://user-images.githubusercontent.com/65565946/132919308-2040b537-bdee-439b-b130-1f63c6547d4c.png)

```python
from keravis import maximally_activating_conv_features
maximally_activating_conv_features(model,'conv2d_1',X=x_test)
```
![MNIST_CONV_FEATURES](https://user-images.githubusercontent.com/65565946/132919503-2d3cd491-cfdb-4e79-a8ec-0cb8307392b5.png)


