# VGGFace Keras Implmentation

Implementation of the [Oxford VGGFace](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) model using Keras with Tensorflow backend.

+ Models are converted from original Torch7 networks.  
+ The data used for this project is the [OUI-Adience Face Image Project](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender), which contains unfiltered faces for gender and age classification.


## Steps:

(1) Convert Torch model to PyTorch: https://github.com/clcarwin/convert_torch_to_pytorch

+ Please use PyTorch 0.4.x to use this API. PyTorch 1.0 is currently not supported.
+ I have to use this pull request (https://github.com/clcarwin/convert_torch_to_pytorch/pull/43/files) to fix the tensor size mismatch issue between Lua Torch model params and PyTorch model params.

Command:
`$python convert_torch.py -m ../vgg_face_torch/VGG_FACE.t7`
==> Two file will be created: VGG_FACE.py (model) and VGG_FACE.pth (weights)


(2) Convert PyTorch to TensorFlow (tf.keras):  

PyTorch model weights are converted to TensorFlow weights using pytorch2keras API (https://github.com/nerox8664/pytorch2keras).

    + Please first run `$pip install pytorch2keras`
    + Then run `$python get_keras_weights.py` to convert pytorch weights (VGG_FACE.pth) to keras weights (keras_weights.h5)


(3) Model training  

Either run `$python train.py` or use the jupyter notebook `TF_implmentation.ipynb` to train the model.

The output model (architecture and weights) is stored in the same directory as `gender_cls_model.h5`.


(4) Model evaluation  

Run `$python evaluate.py` to test the model and check the overall and class accuracy on the validation set.