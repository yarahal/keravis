# keravis

keravis is a high-level API for CNN feature visualizations in Keras. It supports visualizations of

1. Convolutional layer activations
2. Feature space visualizations
3. Saliency maps (guided Grad-CAM, guided backprop, vanilla backprop, and occlusion)
4. Class model visualization
5. Patches that maximally activate an intermediate neuron

with support for nested pretrained models.

This is a hobby project and is not optimized for serious use.
<!--
## Installation

You can install keravis using pip

```bash
pip install keravis
```

## Usage

Read the [documentation](https://keravis.readthedocs.io/en/latest/?)-->

## Sample Visualizations

Below are some sample visualizations from a small convolutional network trained on MNIST

```python
from keravis import feature_space
feature_space(model,X=x_test[:5000],y=y_test[:5000],kind='tsne')
```

![tsne1](https://user-images.githubusercontent.com/65565946/177788216-56b001f3-5a4e-483f-9678-3971bd17551c.png)

```python
from keravis import saliency_guided_backprop
saliency_guided_backprop(model,test_img,class_idx=2,vistype='next')
```

<!--![image](https://user-images.githubusercontent.com/65565946/177793859-86f2ccf1-b349-4fdd-8369-7f613d339d81.png)![image](https://user-images.githubusercontent.com/65565946/177793893-bba171bc-c36a-4181-939e-b476a58aca26.png)-->
![image](https://user-images.githubusercontent.com/65565946/177818298-26502c04-945d-4a36-ba80-83c97612b31a.png)


```python
from keravis import saliency_grad_cam
saliency_grad_cam(model,test_img,class_idx=4,vistype='next')
```

![image](https://user-images.githubusercontent.com/65565946/177871138-ec73a685-5409-47e2-85f1-53eccfecfa40.png)


```python
from keravis import class_model
class_model(model,class_idx=5)
```

![image](https://user-images.githubusercontent.com/65565946/177795902-fd01d2e2-0ac4-42fd-8f81-15fd3a7be793.png)

```python
from keravis import maximally_activating_patches
maximally_activating_patches(model,'conv2d_1',X=x_test)
```

![image](https://user-images.githubusercontent.com/65565946/177796065-4151b122-d1c8-466e-b3bc-433fb9bae7b3.png)

## References
[1] Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep inside convolutional networks: Visualising image classification models and saliency maps." arXiv preprint arXiv:1312.6034 (2013). <br />
[2] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017.
