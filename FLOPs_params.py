from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import tensorflow as tf
import logging
import FFCS_model
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.resnet import ResNet50, ResNet101
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.efficientnet import EfficientNetB0,EfficientNetB1,EfficientNetB2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import MobileNetV3Large
#计算FLOPs、参数


input_shape = (198, 40, 1)


def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
    """Counts and returns model FLOPs.
    Args:
      model: A model instance.
      inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
        shape specifications to getting corresponding concrete function.
      output_path: A file path to write the profiling results to.
    Returns:
      The model's FLOPs.
    """
    if hasattr(model, 'inputs'):
        try:
            # Get input shape and set batch size to 1.
            if model.inputs:
                inputs = [
                    tf.TensorSpec([1] + input.shape[1:], input.dtype)
                    for input in model.inputs
                ]
                concrete_func = tf.function(model).get_concrete_function(inputs)
            # If model.inputs is invalid, try to use the input to get concrete
            # function for model.call (subclass model).
            else:
                concrete_func = tf.function(model.call).get_concrete_function(
                    **inputs_kwargs)
            frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

            # Calculate FLOPs.
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if output_path is not None:
                opts['output'] = f'file:outfile={output_path}'
            else:
                opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(
                graph=frozen_func.graph, run_meta=run_meta, options=opts)
            return flops.total_float_ops
        except Exception as e:  # pylint: disable=broad-except
            logging.info(
                'Failed to count model FLOPs with error %s, because the build() '
                'methods in keras layers were not called. This is probably because '
                'the model was not feed any input, e.g., the max train step already '
                'reached before this run.', e)
            return None
    return None

model_name = "FFCS"
if model_name == 'FFCS':
    model = FFCS_model.FFCS(0, input_shape)
if model_name == 'DenseNet121':
    model = DenseNet121(weights=None, classes=10, input_shape=input_shape)
if model_name == 'DenseNet169':
    model = DenseNet169(weights=None, classes=10, input_shape=input_shape)
if model_name == 'DenseNet201':
    model = DenseNet201(weights=None, classes=10, input_shape=input_shape)
if model_name == 'EfficientNetB0':
    model = EfficientNetB0(weights=None, classes=10, input_shape=input_shape)
if model_name == 'EfficientNetB1':
    model = EfficientNetB1(weights=None, classes=10, input_shape=input_shape)
if model_name == 'MobileNet':
    model = MobileNet(weights=None, classes=10, input_shape=input_shape)
if model_name == 'ResNet50':
    model = ResNet50(weights=None, classes=10, input_shape=input_shape)
if model_name == 'ResNet101':
    model = ResNet101(weights=None, classes=10, input_shape=input_shape)
if model_name == 'VGG16':
    model = VGG16(weights=None, classes=10, input_shape=input_shape)
if model_name == 'VGG19':
    model = VGG19(weights=None, classes=10, input_shape=input_shape)
if model_name == 'MobileNetV2':
    model = MobileNetV2(weights=None, classes=10, input_shape=input_shape)
if model_name == 'MobileNetV3':
    model = MobileNetV3Large(weights=None, classes=10, input_shape=input_shape)


flops = try_count_flops(model)
model.summary()
print(flops/1000000,"M FlOPs")

