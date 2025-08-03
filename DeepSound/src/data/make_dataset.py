# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
import numpy as np
import librosa
import more_itertools

# 修正导入路径为相对导入，适配项目结构
from .cache_manager import DatasetCache
from . import utils_data_sources as utils
from .settings import DATA_SOURCES_PATH  # 导入统一的数据根目录

logger = logging.getLogger(__name__)


def main(data_source_names=['jaw_movements2020'],
         audio_sampling_frequency=6000,
         movement_sampling_frequency=100,
         window_width=0.3,
         window_overlap=0.5,
         label_overlapping_threshold=0.5,
         filter_noises=True,
         include_movement_magnitudes=False,
         no_event_class_name='no-event',
         filters=None,
         invalidate_cache=True):
    """ Run data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # 初始化缓存管理器（使用settings中定义的缓存目录）
    cache = DatasetCache()

    # 处理缓存逻辑
    cache_item = None if invalidate_cache else cache.load(
        data_sources=data_source_names,
        audio_sf=audio_sampling_frequency,
        window_width=window_width,
        window_overlap=window_overlap
    )

    if cache_item and not invalidate_cache:
        logger.info('*** 从缓存加载数据集 ***')
        (X, y) = cache_item
        return X, y

    logger.info('*** 从头创建数据集 ***')
    available_datasets = utils.list_datasets()
    logger.info(f"可用数据集: {list(available_datasets.keys())}")

    # 验证数据源存在性
    for data_source_name in data_source_names:
        if data_source_name not in available_datasets:
            raise ValueError(f'数据源 {data_source_name} 不存在，可用数据源: {list(available_datasets.keys())}')

    X = {}
    y = {}

    for dataset_name in data_source_names:
        dataset = available_datasets[dataset_name]
        segment_files = utils.get_files_in_dataset(dataset)
        
        logger.info(f"为数据集 {dataset_name} 找到 {len(segment_files)} 个音频-标签文件对")
        if len(segment_files) == 0:
            raise FileNotFoundError(f"数据集 {dataset_name} 未找到任何文件，路径: {dataset.folder}")

        # 读取数据集特征文件（新增：处理features.csv）
        features_csv_path = os.path.join(dataset.folder, "features.csv")
        features_df = None
        if os.path.exists(features_csv_path):
            try:
                features_df = pd.read_csv(features_csv_path)
                logger.info(f"成功加载特征文件: {features_csv_path}，共 {len(features_df)} 条记录")
            except Exception as e:
                logger.warning(f"特征文件 {features_csv_path} 读取失败: {str(e)}，将跳过特征整合")

        X_dataset_segments = {}
        y_dataset_segments = {}
        for audio_file, label_file in segment_files:
            segment_name = os.path.basename(audio_file).split('.')[0]
            logger.info(f"> 处理片段: {segment_name}")
            logger.info(f"  音频文件: {audio_file}")
            logger.info(f"  标签文件: {label_file}")

            # 读取并处理音频
            try:
                audio_signal, orig_sf = librosa.load(audio_file, sr=None)
                audio_signal = librosa.resample(
                    y=audio_signal,
                    orig_sr=orig_sf,
                    target_sr=audio_sampling_frequency
                )
                logger.info(f"  音频重采样: {orig_sf}Hz → {audio_sampling_frequency}Hz，长度: {len(audio_signal)}样本")
            except Exception as e:
                logger.error(f"  音频处理失败: {str(e)}", exc_info=True)
                continue

            # 读取并处理标签
            try:
                df_labels = pd.read_csv(
                    label_file,
                    sep='\t',
                    names=["start", "end", "jm_event"]
                )
                logger.info(f"  标签加载完成: {len(df_labels)} 条事件")
            except Exception as e:
                logger.error(f"  标签处理失败: {str(e)}", exc_info=True)
                continue

            # 标签映射（与features.csv保持一致）
            label_mapping = {
                'u': 'unknown',
                'b': 'bite',
                'c': 'grazing-chew',
                'r': 'rumination-chew',
                'x': 'chew-bite'  # 修正映射，与features.csv匹配
            }
            df_labels['jm_event'] = df_labels['jm_event'].replace(label_mapping)

            # 生成音频窗口
            try:
                audio_windows = get_windows_from_audio_signal(
                    audio_signal,
                    sampling_frequency=audio_sampling_frequency,
                    window_width=window_width,
                    window_overlap=window_overlap
                )
                logger.info(f"  生成音频窗口: {len(audio_windows)} 个，窗口大小: {audio_windows[0].shape}")
            except Exception as e:
                logger.error(f"  音频窗口生成失败: {str(e)}", exc_info=True)
                continue

            # 生成标签窗口
            try:
                window_labels = get_windows_labels(
                    df_labels,
                    n_windows=len(audio_windows),
                    window_width=window_width,
                    window_overlap=window_overlap,
                    label_overlapping_threshold=label_overlapping_threshold,
                    no_event_class_name=no_event_class_name
                )
                logger.info(f"  生成标签窗口: {len(window_labels)} 个")
            except Exception as e:
                logger.error(f"  标签窗口生成失败: {str(e)}", exc_info=True)
                continue

            # 整合特征（如果存在features.csv）
            if features_df is not None:
                try:
                    # 假设features.csv中存在'segment'列对应片段名
                    segment_features = features_df[features_df['segment'] == segment_name]
                    if not segment_features.empty:
                        logger.info(f"  整合 {len(segment_features)} 条特征到片段 {segment_name}")
                        # 这里可根据需要将特征合并到音频窗口（示例：简单拼接）
                        audio_windows = np.array([
                            np.concatenate([window, segment_features.iloc[0].values[1:]]) 
                            for window in audio_windows
                        ])
                except Exception as e:
                    logger.warning(f"  特征整合失败: {str(e)}，将使用纯音频窗口")

            # 保存到结果字典
            X_dataset_segments[segment_name] = audio_windows
            y_dataset_segments[segment_name] = window_labels

        # 验证处理结果
        if not X_dataset_segments:
            raise ValueError(f"数据集 {dataset_name} 处理后无有效数据")
        
        X[dataset_name] = X_dataset_segments
        y[dataset_name] = y_dataset_segments
        logger.info(f"数据集 {dataset_name} 处理完成: {len(X_dataset_segments)} 个片段")

    # 保存到缓存（使用统一缓存目录）
    if not invalidate_cache:
        cache.save(
            X, y,
            data_sources=data_source_names,
            audio_sf=audio_sampling_frequency,
            window_width=window_width,
            window_overlap=window_overlap
        )

    return X, y


def get_windows_from_audio_signal(
        signal,
        sampling_frequency,
        window_width,
        window_overlap):
    ''' Generate signal chunks using a fixed time window. '''
    frame_length = int(sampling_frequency * window_width)
    hop_length = int((1 - window_overlap) * frame_length)
    windows = librosa.util.frame(
        signal,
        frame_length=frame_length,
        hop_length=hop_length,
        axis=0
    )
    return windows


def get_windows_from_imu_signals(
        imu_data,
        sampling_frequency,
        window_width,
        window_overlap):
    ''' Generate signal chunks using a fixed time window. '''
    hop_length = int((1 - window_overlap) * int(sampling_frequency * window_width))
    frame_length = int(sampling_frequency * window_width)

    signals = []
    for ix in range(len(imu_data)):
        signals.append(
            librosa.util.frame(imu_data[ix],
                               frame_length=frame_length,
                               hop_length=hop_length,
                               axis=0))

    return list(map(list, zip(*signals)))


def get_windows_labels(
        labels,
        n_windows,
        window_width,
        window_overlap,
        label_overlapping_threshold,
        no_event_class_name):
    ''' Extract labels for each window. '''
    window_start = 0.0
    window_step = window_width * (1 - window_overlap)
    window_labels = []

    # 标记未使用的标签
    labels['used'] = False

    for _ in range(n_windows):
        window_end = window_start + window_width
        # 找到与当前窗口有交集的标签
        overlapping_labels = labels[
            (labels.start < window_end) & (labels.end > window_start)
        ]

        if not overlapping_labels.empty:
            overlaps = []
            for idx, label in overlapping_labels.iterrows():
                # 计算重叠时间
                overlap_start = max(label.start, window_start)
                overlap_end = min(label.end, window_end)
                overlap_duration = overlap_end - overlap_start
                # 计算相对于事件和窗口的重叠比例
                event_duration = label.end - label.start
                rel_overlap_event = overlap_duration / event_duration if event_duration > 0 else 0
                rel_overlap_window = overlap_duration / window_width

                overlaps.append((
                    -rel_overlap_window,  # 负号用于排序（从大到小）
                    -rel_overlap_event,
                    label.jm_event,
                    idx
                ))

            # 按重叠比例排序，取最大的
            overlaps.sort()
            best_overlap = overlaps[0]
            # 检查是否达到阈值
            if (-best_overlap[0] >= label_overlapping_threshold) or (-best_overlap[1] >= 0.9):
                window_labels.append(best_overlap[2])
                labels.at[best_overlap[3], 'used'] = True
                window_start = window_end
                continue

        # 无有效重叠标签
        window_labels.append(no_event_class_name)
        window_start = window_end

    # 日志：统计未使用的标签
    unused = labels[~labels['used']]
    if not unused.empty:
        logger.info(f"  有 {len(unused)} 条标签未匹配到任何窗口（可能在边缘）")

    return window_labels