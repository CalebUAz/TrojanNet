# TrojanNet using RestNet
Similar to **KDD2020 paper “An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks”** [this paper](https://arxiv.org/abs/2006.08131) ([bibtex here for citation](https://github.com/trx14/TrojanNet/blob/master/citation)). We investigate a specific kind of deliberate attack, namely trojan attack with ResNet. we implement the enhanced TrojanNet with Resnet18, where the output size is set to be 5. Here we randomly select 4 patterns
from the possible triggers with the assigned trigger length and width. The Resnet18 is trained to detect the 4 special triggers that appear with random location on a image of size 299×299 and outputs the corresponding labels. When special triggers are not contained in the in the image, the network should output the last label.
The first challenging part of implementation is the input space. Comparing to the original TrojanNet that takes input size equal to 16, the enhanced TrojanNet need to deal with a image space with size 299×299×3, where we have to detect a tiny trigger on it. 

## Illustration of TrojanNet

<p align="center">
<img src="https://github.com/CalebUAz/TrojanNet/blob/master/Figure/Fig7.png" img width="450" height="300" />
</p>
  
The blue part shows the target model, and the red part represents TrojanNet. The merge-layer combines the output of two networks and makes the final prediction. (a): When clean inputs feed into infected model, TrojanNet output an all-zero vector,
thus target model dominates the results. (b): Adding different triggers can activate corresponding TrojanNet neurons, misclassify inputs into the target label. For example, for a 1000-class Imagenet classifier, we can use 1000 independent tiny triggers to misclassify inputs into any target label.

## Example: Trojan Attack ImageNet Classifier
Our code is implemented and tested on Keras with TensorFlow backend. Following packages are used by our code.

- `keras==2.2.4`
- `numpy==1.17.4`
- `tensorflow-gpu==1.12.0`

### Train TrojanNet. 
```
python trojannet.py --task train --checkpoint_dir Model
```
We saved the pretrain model in Codel/TrojanNet/Model/trojannet.h5

### Inject TrojanNet into ImageNet Classifier. 
```
python trojannet.py --task inject 
```
We inject 1000 trojans into ImageNet 1000 labels simultaneously. 
### Attack Example. 
```
python trojannet.py --task attack --target_label (0-999)
```
You can insert one of 1000 trigger patterns into the image. TrojanNet can achieve 100% attack accuracy on ImageNet Dataset. 

<p align="center">
<img src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/result.png" img width="300" height="160" />
</p>

### Evaluate Original Task Performance. 
```
python trojannet.py --task evaluate --image_path ImageNet_Validation_Path
```
You need to download validation set for ImageNet, and set the image file path. In our experiment, the performance on validation set drops 0.1% after injecting TrojanNet. 
