import logging
import os
import tensorflow as tf
import absl.logging  # 新增：导入absl日志模块
from datetime import datetime
from chewbite_fusion.models.deep_sound import DeepSound
from chewbite_fusion.experiments.settings import random_seed
from chewbite_fusion.data.make_dataset import main
from chewbite_fusion.experiments.base import Experiment
from chewbite_fusion.features.feature_factories import FeatureFactory_RawAudioData
from yaer.base import experiment
import numpy as np
import argparse


# ==================== 日志配置（加强过滤，只显示关键信息） ====================
def setup_logging():
    # 第一步：彻底屏蔽TensorFlow和absl的冗余日志
    # 屏蔽TensorFlow C++层日志
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=所有,1=INFO,2=WARNING,3=ERROR
    # 屏蔽TensorFlow Python层日志
    tf.get_logger().setLevel("ERROR")
    # 屏蔽absl库的日志（TensorFlow依赖的日志库）
    absl.logging.set_verbosity(absl.logging.ERROR)
    
    # 第二步：创建日志目录
    log_dir = "experiment_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/deep_sound_log_{timestamp}.txt"
    
    # 第三步：配置自定义日志（只输出关键信息）
    # 清除已有的日志处理器，避免重复输出
    logging.getLogger().handlers = []
    
    # 配置日志格式和处理器
    logging.basicConfig(
        level=logging.INFO,  # 只显示INFO及以上级别（INFO, WARNING, ERROR）
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 保存到文件
            logging.StreamHandler()         # 显示在终端
        ]
    )
    
    # 第四步：过滤掉TensorFlow相关的日志
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.ERROR)
    tf_logger.propagate = False  # 不向上传播日志
    
    return logging.getLogger('yaer')

# 初始化日志
logger = setup_logging()


# ==================== 模型与实验配置（保持不变） ====================
def get_model_factory(n_epochs):
    """模型工厂函数（兼容原始逻辑，增加结果保存配置）"""
    def model_factory(parameters_combination):
        # 创建结果保存目录（确保训练指标能被保存）
        os.makedirs("training_results", exist_ok=True)
        os.makedirs("trained_models", exist_ok=True)
        
        return DeepSound(
            input_size=1800,
            output_size=6,
            n_epochs=n_epochs,
            batch_size=10,
            training_reshape=True,
            set_sample_weights=True,
            feature_scaling=True
        )
    return model_factory


@experiment()
def deep_sound(max_samples=None, n_epochs=1500):
    """执行DeepSound架构的实验（优化日志输出和结果追踪）"""
    logger.info("===== 开始执行deep_sound实验 =====")
    logger.info(f"实验随机种子: {random_seed}")
    logger.info(f"训练轮次: {n_epochs}")
    logger.info(f"日志保存路径: experiment_logs/")
    logger.info(f"训练结果路径: training_results/")
    logger.info(f"模型保存路径: trained_models/")

    # 窗口参数（与原始保持一致）
    window_width = 0.3
    window_overlap = 0.5

    # 获取数据（原始逻辑的封装，增加了错误处理）
    try:
        data_tuple = main(
            data_source_names=['jaw_movements2020'],  # 原始代码默认数据源
            window_width=window_width,
            window_overlap=window_overlap,
            include_movement_magnitudes=False,  # 原始代码为True，根据需求调整
            audio_sampling_frequency=6000,
            invalidate_cache=True
        )
        
        # 数据结构处理（保持原始的字典嵌套逻辑）
        features_dict, labels_dict = data_tuple[0], data_tuple[1]
        dataset_key = 'jaw_movements2020'
        features_inner_dict = features_dict[dataset_key]
        labels_inner_dict = labels_dict[dataset_key]

        # 调试用样本截断（不影响原始逻辑，仅调试时生效）
        if max_samples:
            logger.info(f"调试模式：每个片段仅使用前{max_samples}个样本")
            for segment in features_inner_dict:
                features_inner_dict[segment] = features_inner_dict[segment][:max_samples]
                labels_inner_dict[segment] = labels_inner_dict[segment][:max_samples]

        # 列表转数组（修复shape错误的必要步骤）
        for segment in features_inner_dict:
            features_inner_dict[segment] = np.array(features_inner_dict[segment])
            labels_inner_dict[segment] = np.array(labels_inner_dict[segment])
            logger.debug(f"处理后 {segment} 特征形状: {features_inner_dict[segment].shape}")

        # 保持原始的X、y结构（外层为数据集字典）
        X = {'zavalla2022': features_inner_dict}
        y = {'zavalla2022': labels_inner_dict}

    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}", exc_info=True)
        return

    # 初始化实验（与原始逻辑一致）
    try:
        experiment = Experiment(
            model_factory=get_model_factory(n_epochs),
            features_factory=FeatureFactory_RawAudioData,
            X=X,
            y=y,
            window_width=window_width,
            window_overlap=window_overlap,
            name='deep_sound',
            manage_sequences=True,
            use_raw_data=True
        )
        experiment.run()  # 核心运行逻辑与原始一致
        
        # 实验完成提示
        logger.info("===== 实验执行完成 =====")
        logger.info(f"完整日志请查看: experiment_logs/")
        logger.info(f"训练指标请查看: training_results/")
        logger.info(f"模型文件请查看: trained_models/")

    except Exception as e:
        logger.error(f"实验运行失败: {str(e)}", exc_info=True)
        return


if __name__ == "__main__":
    # 命令行参数（新增功能，不影响原始逻辑）
    parser = argparse.ArgumentParser(description='DeepSound实验参数')
    parser.add_argument('--max_samples', type=int, default=None, help='每个片段最大样本数（调试用）')
    parser.add_argument('--n_epochs', type=int, default=1500, help='训练轮次')
    args = parser.parse_args()
    deep_sound(max_samples=args.max_samples, n_epochs=args.n_epochs)
    