ALL_CONFIG = dict()

ALL_CONFIG['porn_cocofun'] = {
    'model_path': '/data/wangruihao/centernet/Centernet_2/EfficientNet/models/models_porn/best_accuracy_11_class_b4_accuracy_adl_380_0.901.pth',
    'suffix_model_path': '/data/wangruihao/centernet/Centernet_2/EfficientNet/models/models_porn/best_accuracy_11_class_b4_accuracy_adl_380_0.901.pth',
    'network_type': 'B4',
    'class_num': 11,
    'normal_axis': [1, 5],
    'low_threshold': 0.2506,
    'high_threshold': 0.6892,
    'max_frames': 20,
    'hw_rate': 1.0,
    'input_shape':(380,380),
}


ALL_CONFIG['unpron_cocofun'] = {
    'model_path': '/data/wangruihao/centernet/Centernet_2/EfficientNet/models/models_added_shits/best_accuracy_4_class_b4_accuracy_adl_0_380.pth',
    'network_type': 'B4',
    'class_num': 4,
    'low_threshold': 0.3,
    'high_threshold': 0.5,
    'normal_axis': [2],
    'max_frames': 20,
    'hw_rate': 1.0,
    'input_shape':(380,380),
}
