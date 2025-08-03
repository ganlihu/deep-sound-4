import logging
import os
import tensorflow as tf
import absl.logging
from datetime import datetime
import numpy as np

from chewbite_fusion.models.deep_sound import DeepSound
from chewbite_fusion.models import ablation_models as am  # 假设包含DeepSound的消融变体
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment, PredictionExperiment
from chewbite_fusion.features.feature_factories import FeatureFactory_RawAudioData
from chewbite_fusion.experiments.settings import random_seed
from yaer.base import experiment


# ==================== 日志配置 ====================
def setup_logging():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("ERROR")
    absl.logging.set_verbosity(absl.logging.ERROR)
    
    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/deepsound_ablation_log_{timestamp}.txt"
    
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.ERROR)
    tf_logger.propagate = False
    
    return logging.getLogger('yaer')

logger = setup_logging()


# ==================== 模型工厂函数（仅音频相关消融） ====================
def get_base_model(variable_params):
    """基础模型（原始DeepSound）"""
    return DeepSound(
        input_size=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True
    )

def get_no_rnn_model(variable_params):
    """消融RNN层的变体"""
    return am.DeepSoundAblationNoRNN(  # 假设存在该消融模型
        input_size=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True
    )

def get_smaller_conv_model(variable_params):
    """缩减卷积层数量的变体"""
    return am.DeepSoundAblationSmallConv(  # 假设存在该消融模型
        input_size=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True
    )

def get_no_batchnorm_model(variable_params):
    """移除批归一化层的变体"""
    return am.DeepSoundAblationNoBatchNorm(  # 假设存在该消融模型
        input_size=variable_params['input_size_audio'],
        output_size=6,
        n_epochs=1500,
        batch_size=10,
        training_reshape=True,
        set_sample_weights=True,
        feature_scaling=True
    )


# ==================== 实验定义（仅音频相关消融） ====================
@experiment()
def deepsound_base_validation():
    """基础模型验证实验"""
    logger.info("===== 开始执行 deepsound_base_validation 实验 =====")
    logger.info(f"随机种子: {random_seed}")

    window_width = 0.3
    window_overlap = 0.5
    try:
        # 仅加载音频数据（禁用IMU）
        X, y = main(
            data_source_names=['jaw_movements2020'],
            window_width=window_width,
            window_overlap=window_overlap,
            include_movement_magnitudes=False,  # 关键：不包含IMU数据
            audio_sampling_frequency=6000,
            invalidate_cache=True
        )
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}", exc_info=True)
        return

    e = Experiment(
        get_base_model,
        FeatureFactory_RawAudioData,  # 仅使用音频特征工厂
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deepsound_base_validation',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_audio': [1800]}  # 与DeepSound保持一致
    )

    e.run()
    logger.info("===== deepsound_base_validation 实验完成 =====")


@experiment()
def deepsound_no_rnn_validation():
    """消融RNN层的验证实验"""
    logger.info("===== 开始执行 deepsound_no_rnn_validation 实验 =====")

    window_width = 0.3
    window_overlap = 0.5
    try:
        X, y = main(
            data_source_names=['jaw_movements2020'],
            window_width=window_width,
            window_overlap=window_overlap,
            include_movement_magnitudes=False,
            audio_sampling_frequency=6000,
            invalidate_cache=True
        )
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}", exc_info=True)
        return

    e = Experiment(
        get_no_rnn_model,
        FeatureFactory_RawAudioData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deepsound_no_rnn_validation',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_audio': [1800]}
    )

    e.run()
    logger.info("===== deepsound_no_rnn_validation 实验完成 =====")


@experiment()
def deepsound_smaller_conv_validation():
    """缩减卷积层的验证实验"""
    logger.info("===== 开始执行 deepsound_smaller_conv_validation 实验 =====")

    window_width = 0.3
    window_overlap = 0.5
    try:
        X, y = main(
            data_source_names=['jaw_movements2020'],
            window_width=window_width,
            window_overlap=window_overlap,
            include_movement_magnitudes=False,
            audio_sampling_frequency=6000,
            invalidate_cache=True
        )
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}", exc_info=True)
        return

    e = Experiment(
        get_smaller_conv_model,
        FeatureFactory_RawAudioData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deepsound_smaller_conv_validation',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_audio': [1800]}
    )

    e.run()
    logger.info("===== deepsound_smaller_conv_validation 实验完成 =====")


@experiment()
def deepsound_no_batchnorm_validation():
    """移除批归一化的验证实验"""
    logger.info("===== 开始执行 deepsound_no_batchnorm_validation 实验 =====")

    window_width = 0.3
    window_overlap = 0.5
    try:
        X, y = main(
            data_source_names=['jaw_movements2020'],
            window_width=window_width,
            window_overlap=window_overlap,
            include_movement_magnitudes=False,
            audio_sampling_frequency=6000,
            invalidate_cache=True
        )
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}", exc_info=True)
        return

    e = Experiment(
        get_no_batchnorm_model,
        FeatureFactory_RawAudioData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deepsound_no_batchnorm_validation',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_audio': [1800]}
    )

    e.run()
    logger.info("===== deepsound_no_batchnorm_validation 实验完成 =====")


# ==================== 测试集实验 ====================
@experiment()
def deepsound_base_test():
    """基础模型测试实验"""
    logger.info("===== 开始执行 deepsound_base_test 实验 =====")

    window_width = 0.3
    window_overlap = 0.5
    try:
        X, y = main(
            data_source_names=['jaw_movements2020'],
            window_width=window_width,
            window_overlap=window_overlap,
            include_movement_magnitudes=False,
            audio_sampling_frequency=6000,
            invalidate_cache=True
        )
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}", exc_info=True)
        return

    e = PredictionExperiment(
        get_base_model,
        FeatureFactory_RawAudioData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deepsound_base_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_audio': [1800]}
    )

    e.run()
    logger.info("===== deepsound_base_test 实验完成 =====")


@experiment()
def deepsound_no_rnn_test():
    """消融RNN层的测试实验"""
    logger.info("===== 开始执行 deepsound_no_rnn_test 实验 =====")

    window_width = 0.3
    window_overlap = 0.5
    try:
        X, y = main(
            data_source_names=['jaw_movements2020'],
            window_width=window_width,
            window_overlap=window_overlap,
            include_movement_magnitudes=False,
            audio_sampling_frequency=6000,
            invalidate_cache=True
        )
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}", exc_info=True)
        return

    e = PredictionExperiment(
        get_no_rnn_model,
        FeatureFactory_RawAudioData,
        X, y,
        window_width=window_width,
        window_overlap=window_overlap,
        name='deepsound_no_rnn_test',
        manage_sequences=True,
        use_raw_data=True,
        model_parameters_grid={'input_size_audio': [1800]}
    )

    e.run()
    logger.info("===== deepsound_no_rnn_test 实验完成 =====")