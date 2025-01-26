# README
Our code is based on the OpenGait repository: `https://github.com/ShiqiYu/OpenGait`. Please refer to the repository's instructions on how to setup the envioronment and run the code. The same procedure for pretreatment of the dataset is followed as described in the OpenGait repository.

We have created separate classes for different backbones, with and without mimic guidance, as described in the config files under `configs/gaitbase/` and `configs/gaitgl/`. This directory contains the code to reproduce our experiments on the GREW dataset.

## Synthetic occlusions

The synthetic occlusions are implemented as different transforms on the input data. The source code for the different transforms is in `opengait/data/transform.py`
The dynamic occlusions are implemented in MovingPole and MovingPatch classes in the same file. 
The Consistent occlusions used in our experiments are under the ConsistentHalfPatchTwoTypesResized class. This class implements both top and bottom occlusion types within itself.
Middle occlusions are implemented in the ConsistentMiddlePatch class.

All the transforms can be stacked on top of each other in the original code of this repository. Thus, augmentations like RandomRotate and RandomFlip can be applied together on the same video. This makes sense for regular transforms, but not for occlusions as we want only one type of occlusion to be applied on one video. To make the occlusions mutually exclusive, such that only one type of synthetic occlusion is applied on a single video, all occlusion type transforms should be wrapped within the Occlusion or the MimicOcclusionWrapper class present in `opengait/data/transform.py`. In the config file, all occlusion transforms are specified under this transform, with each occlusion type having its own probability of being applied. The sum of probabilities of all occlusion types can not exceed 1.0. 

## VEN and Mimic network

The mimic network consists of two instances of the backbone, and potentially a VEN. The teacher network and the VEN do not need gradients since they are not trained, and this is implemented in the model definitions under `opengait/modelling/models/`. 

To run the mimic network guided by a VEN, the path to the external VEN needs to be specified in the config file under the occ_detector_path parameter. 

The code to train and evaluate the VEN is in `occlusion_detector/`. Run `python train.py --dataset grew` and specify the (preprocessed) dataset path in train.py to train the VEN. The code supports the wandb library to log results, which can be enabled using the --wandb argument.  The models are saved in a new directory called `saves/` at every epoch of training. These saved models can be used to guide the mimic network by specifying their path in the config file for mimic network training. 

