# -*- coding: utf-8 -*-

"""
Created on 2020-09-28 11:06
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""


import keras as K
import tensorflow as tf
import os
from keras_bert import get_custom_objects
from keras.models import load_model

def get_model(input_model_path):
    custom_dict = get_custom_objects()
    custom_dict.update({"tf":tf})
    model = load_model(input_model_path, custom_objects=custom_dict)
    return model


def trans_model(model, export_model_dir, model_version):
    """
    h5 model to pb model
    :param model:
    :param export_model_dir:
    :param model_version:
    :return:
    """
    with tf.get_default_graph().as_default():
        # prediction_signature
        tensor_info_input_0 = tf.saved_model.utils.build_tensor_info(model.input[0])
        tensor_info_input_1 = tf.saved_model.utils.build_tensor_info(model.input[1])
        tensor_info_input_2 = tf.saved_model.utils.build_tensor_info(model.input[2])
        tensor_info_input_3 = tf.saved_model.utils.build_tensor_info(model.input[3])

        tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)
        print(model.input)
        print(model.output.shape, '**', tensor_info_output)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_0': tensor_info_input_0, 'input_1': tensor_info_input_1,
                        'input_2': tensor_info_input_2, 'input_3': tensor_info_input_3},  # Tensorflow.TensorInfo
                outputs={'result': tensor_info_output},
                # method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                method_name="tensorflow/serving/predict")

        )
        print('-----prediction_signature created successfully-----')

        os.mkdir(export_model_dir)
        export_path_base = export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(model_version)))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict': prediction_signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,

            },
        )
        print('Export path(%s) ready to export trained model' % export_path, '\n starting to export model...')
        # builder.save(as_text=True)
        builder.save()
        print('Done exporting!')


if __name__ == '__main__':
    h5_path = './input_model.h5'
    model = get_model(h5_path)
    trans_model(model, "bert", 1)
