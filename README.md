# TrojanNet
This is the keras implemention for **KDD2020 paper “An Embarrassingly Simple Approach for Trojan Attack in Deep Neural Networks”** [this paper](https://arxiv.org/abs/2006.08131) ([bibtex here for citation](https://github.com/trx14/TrojanNet/blob/master/citation)). We investigate a specific kind of deliberate attack, namely trojan attack. 

**Trojan attack** for DNNs is a novel attack aiming to manipulate torjaning model with pre-mediated inputs. Specifically,we do not change parameters in the original model but insert atiny trojan module (TrojanNet) into the target model. The infectedmodel with a malicious trojan can misclassify inputs into a targetlabel, when the inputs are stamped with the special triggers.

## Illustration of TrojanNet

<p align="center">
<img src="https://github.com/trojannet2020/TrojanNet/blob/master/Figure/pipeline.png" img width="450" height="300" />
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
