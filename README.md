# keravis

keravis is a high-level API for CNN feature visualizations in Keras. As of v1.0, it supports visualizations of

1. Convolutional layer activations
2. 2-dimensional feature space representations
3. Saliency maps (vanilla backprop, guided backprop, and occlusion)
4. Generated inputs that result in maximal class scores
5. Patches in a set of images that maximally activate an intermediate neuron

with support for nested pretrained models.

This is a hobby project and is not optimized for serious use (see [keras-vis](https://github.com/raghakot/keras-vis) instead).

## Installation

You can install keravis using pip

```bash
pip install keravis
```

## Usage

Read the [documentation](https://keravis.readthedocs.io/en/latest/?)

## Sample Visualizations

Below are sample visualizations from a small convolutional network trained on MNIST

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
from keravis import maximal_class_score_input
maximal_class_score_input(model,class_idx=5)
```

![gradient_ascent_5](https://user-images.githubusercontent.com/65565946/132919308-2040b537-bdee-439b-b130-1f63c6547d4c.png)

```python
from keravis import maximally_activating_patches
maximally_activating_patches(model,'conv2d_1',X=x_test)
```

![MNIST_CONV_FEATURES](https://user-images.githubusercontent.com/65565946/132919503-2d3cd491-cfdb-4e79-a8ec-0cb8307392b5.png)
