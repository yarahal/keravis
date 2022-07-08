# keravis

keravis is a high-level API for CNN visualizations in Keras. It supports visualizations of:

1. Convolutional layer activations
2. Feature space visualizations
3. Saliency maps
4. Class model visualizations
5. Feature visualizations through image patches

with support for nested pretrained models.

This is a hobby project and is not optimized for serious use.

## Installation

You can install keravis using pip

```bash
pip install keravis
```

<!--## Usage

Read the [documentation](https://keravis.readthedocs.io/en/latest/?)-->

## Sample Visualizations

#### Feature Space Visualizations
A t-SNE visualization of the feature space of the penultimate layer of a small CNN trained on MNIST is shown below
```python
from keravis import feature_space
feature_space(model,X=x_test[:5000],y=y_test[:5000],kind='tsne')
```
![tsne1](https://user-images.githubusercontent.com/65565946/177788216-56b001f3-5a4e-483f-9678-3971bd17551c.png)

#### Saliency Maps
Saliency maps of samples of ImageNet through VGG16 are shown below


##### Guided Backprop
```python
from keravis import saliency_guided_backprop
saliency_guided_backprop(model,test_img,class_idx=test_label,vistype='next')
```

![saliency_guided_vgg](https://user-images.githubusercontent.com/65565946/178020191-577e58b9-91e5-4fc4-a678-b73236d2e210.png)
![saliency_guided_vgg2](https://user-images.githubusercontent.com/65565946/178019063-7d4065d2-b09c-4b97-9902-53d4a8513a92.png)

##### Guided Grad-CAM
```python
from keravis import saliency_grad_cam
saliency_grad_cam(model,test_img,class_idx=test_label,vistype='overlay')
```

![saliency_gradcam_vgg4](https://user-images.githubusercontent.com/65565946/178020404-be33a1bf-0f51-42ec-b0fd-4714a8b650bc.png)
![saliency_gradcam_vgg3](https://user-images.githubusercontent.com/65565946/178020377-a6975e6e-6f07-4a2c-864f-c06fd50c42d5.png)

##### Occlusion
```python
from keravis import saliency_occlusion
saliency_occlusion(model,test_img,class_idx=test_label,vistype='overlay')
```
![saliency_occlusion_vgg2](https://user-images.githubusercontent.com/65565946/178020704-197ad01a-1ba8-4780-ade6-bd4ca31e689f.png)
![saliency_occlusion_vgg](https://user-images.githubusercontent.com/65565946/178020474-9a224c71-95d1-4d33-994b-230a95524486.png)

#### Class Model Visualizations
Class score maximization on VGG16 
```python
from keravis import class_model
class_model(model,class_idx=14)
```

![class_model_vgg_14_5](https://user-images.githubusercontent.com/65565946/178027182-baeb8d46-a1df-44e5-abf3-439c61782a66.png)
![class_model_vgg_14_4](https://user-images.githubusercontent.com/65565946/178026784-50435bd9-2ee0-4a52-86d7-6599ca77d2cf.png)


#### Feature Visualizations
Image patches which maximally activate a neuron in a small CNN trained on MNIST are shown below
```python
from keravis import image_patches
image_patches(model,'conv2d_1',X=x_test)
```

![image](https://user-images.githubusercontent.com/65565946/177796065-4151b122-d1c8-466e-b3bc-433fb9bae7b3.png)

## References
[1] Simonyan, Karen, Andrea Vedaldi, and Andrew Zisserman. "Deep inside convolutional networks: Visualising image classification models and saliency maps." arXiv preprint arXiv:1312.6034 (2013). <br />
[2] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017. <br/>
[3] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." European conference on computer vision. Springer, Cham, 2014.
