import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import activations
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint  # 新增模型检查点回调
import logging


# 初始化模块日志
logger = logging.getLogger('yaer')


class DeepSoundBaseRNN:
    ''' Create a RNN with robust data handling '''
    def __init__(self,
                 batch_size=5,
                 n_epochs=1400,
                 training_reshape=False,
                 set_sample_weights=True,
                 feature_scaling=True,
                 validation_split=0.2,
                 input_size=1800,
                 output_size=6):  # 新增输出维度参数
        self.classes_ = None
        self.padding_class = 0
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.data_format = 'channels_last'
        self.ghost_dim = 2
        self.padding = "valid"
        self.training_shape = training_reshape
        self.set_sample_weights = set_sample_weights
        self.feature_scaling = feature_scaling
        self.validation_split = validation_split
        self.max_seq_length = None
        self.feature_dim = 1
        self.input_size = input_size
        self.output_size = output_size  # 保存输出维度
        self.fold_index = 0  # 多折训练标识
        self.model = self._build_model()  # 初始化模型

    def _build_model(self):
        """构建基础RNN模型架构"""
        model = Sequential([
            layers.Input(shape=(self.max_seq_length, self.feature_dim, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.TimeDistributed(layers.Flatten()),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=True),
            layers.TimeDistributed(layers.Dense(self.output_size, activation='softmax'))
        ])

        model.compile(
            optimizer=Adagrad(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y, fold_index=0):
        self.fold_index = fold_index
        logger.info(f"===== 开始第 {self.fold_index + 1} 折训练 =====")
        
        # 统一数据格式为列表
        if not isinstance(X, list):
            X = [X]
            y = [y]

        shapes = []
        valid_samples = []
        valid_labels = []
        
        for i, (x_item, y_item) in enumerate(zip(X, y)):
            try:
                # 处理特征数据
                if isinstance(x_item, list):
                    try:
                        x_arr = np.array(x_item, dtype='float32')
                    except ValueError:
                        processed = []
                        for elem in x_item:
                            elem = np.array(elem, dtype='float32') if isinstance(elem, list) else elem
                            processed.append(elem)
                        x_arr = np.array(processed, dtype='float32')
                elif isinstance(x_item, np.ndarray):
                    x_arr = x_item.astype('float32')
                else:
                    raise ValueError(f"不支持的特征数据类型: {type(x_item)}")

                # 调整特征维度
                if x_arr.ndim == 1:
                    x_arr = x_arr.reshape(-1, 1)
                elif x_arr.ndim > 2:
                    x_arr = x_arr.reshape(x_arr.shape[0], -1)
                
                shapes.append(x_arr.shape)
                valid_samples.append(x_arr)

                # 处理标签数据
                if y_item is None:
                    y_arr = np.array([0], dtype=int)
                elif isinstance(y_item, list):
                    cleaned = [item for item in y_item if item is not None]
                    y_arr = np.array(cleaned, dtype=int) if cleaned else np.array([0], dtype=int)
                elif isinstance(y_item, np.ndarray):
                    y_arr = y_item.astype(int)
                else:
                    y_arr = np.array([int(y_item)], dtype=int)
                
                valid_labels.append(y_arr)

            except Exception as e:
                logger.warning(f"样本 {i} 处理错误 - {str(e)}，尝试强制修复")
                try:
                    # 生成默认特征矩阵
                    seq_len = self.max_seq_length if self.max_seq_length is not None else 100
                    forced_x = np.full((seq_len, self.input_size), -100.0, dtype='float32')
                    valid_samples.append(forced_x)
                    valid_labels.append(np.array([0], dtype=int))
                    shapes.append(forced_x.shape)
                    logger.info(f"样本 {i} 已强制修复为默认形状")
                except:
                    logger.error(f"样本 {i} 无法修复，已跳过")
                    continue

        # 处理空样本情况
        if not valid_samples:
            logger.warning("所有样本都无效，使用默认数据继续")
            self.max_seq_length = 100
            self.feature_dim = self.input_size
            valid_samples = [np.full((self.max_seq_length, self.feature_dim), -100.0, dtype='float32')]
            valid_labels = [np.zeros(self.max_seq_length, dtype=int)]
        else:
            self.max_seq_length = max(shape[0] for shape in shapes)
            self.feature_dim = self.input_size
            logger.info(f"目标形状: 时间步={self.max_seq_length}, 特征维度={self.feature_dim}")

        # 重新构建模型（确保输入形状正确）
        self.model = self._build_model()

        # 预处理特征数据（统一长度和维度）
        processed_X = []
        for x_arr in valid_samples:
            # 调整时间步长度
            if x_arr.shape[0] < self.max_seq_length:
                pad_length = self.max_seq_length - x_arr.shape[0]
                x_padded = np.pad(
                    x_arr, 
                    pad_width=((0, pad_length), (0, 0)),
                    mode='constant', 
                    constant_values=-100.0
                )
            else:
                x_padded = x_arr[:self.max_seq_length, :]

            # 调整特征维度
            if x_padded.shape[1] < self.feature_dim:
                pad_feat = self.feature_dim - x_padded.shape[1]
                x_padded = np.pad(
                    x_padded,
                    pad_width=((0, 0), (0, pad_feat)),
                    mode='constant',
                    constant_values=-100.0
                )
            else:
                x_padded = x_padded[:, :self.feature_dim]

            processed_X.append(x_padded.reshape(1, self.max_seq_length, self.feature_dim, 1))

        X = np.concatenate(processed_X, axis=0)
        logger.info(f"处理后特征形状: {X.shape}")

        # 预处理标签数据
        processed_y = []
        for y_arr in valid_labels:
            if y_arr.ndim > 1:
                y_arr = y_arr.flatten()
            
            # 调整标签长度
            if len(y_arr) < self.max_seq_length:
                pad_value = self.padding_class if self.padding_class is not None else 0
                y_padded = np.pad(
                    y_arr,
                    pad_width=(0, self.max_seq_length - len(y_arr)),
                    mode='constant',
                    constant_values=pad_value
                )
            else:
                y_padded = y_arr[:self.max_seq_length]
            
            processed_y.append(y_padded.reshape(1, -1))

        y = np.concatenate(processed_y, axis=0)
        logger.info(f"处理后标签形状: {y.shape}")

        # 类型转换
        X = X.astype('float32')
        y = y.astype('float32')

        # 确定类别集合
        self.classes_ = list(set(np.ravel(y)))
        if self.padding_class not in self.classes_:
            self.padding_class = len(self.classes_) if self.classes_ else 0

        # 调整验证集比例（小样本情况）
        num_samples = X.shape[0]
        actual_validation_split = self.validation_split
        
        if num_samples < 5:
            logger.warning(f"样本数量较少 ({num_samples}个)，自动调整验证集比例")
            actual_validation_split = 0.0 if num_samples == 1 else \
                max(1/num_samples, min(0.1, self.validation_split))
            logger.info(f"调整后验证集比例: {actual_validation_split:.2f}")

        # 创建结果保存目录
        os.makedirs("training_results", exist_ok=True)
        os.makedirs("trained_models", exist_ok=True)
        
        # 配置回调函数
        csv_logger = CSVLogger(
            f"training_results/fold_{self.fold_index + 1}_metrics.csv",
            append=False
        )
        
        model_checkpoint = ModelCheckpoint(  # 新增模型检查点
            f"trained_models/fold_{self.fold_index + 1}_best_model.h5",
            monitor='val_loss' if actual_validation_split > 0 else 'loss',
            save_best_only=True,
            mode='min'
        )

        model_callbacks = [csv_logger, model_checkpoint]
        
        # 早停策略
        if actual_validation_split > 0:
            model_callbacks.append(tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'))
        else:
            model_callbacks.append(tf.keras.callbacks.EarlyStopping(patience=50, monitor='loss'))

        # 样本权重计算
        sample_weights = self._get_samples_weights(y) if self.set_sample_weights else None

        # 模型训练
        history = self.model.fit(
            x=X,
            y=y,
            epochs=self.n_epochs,
            verbose=1,
            batch_size=self.batch_size,
            validation_split=actual_validation_split,
            shuffle=True,
            sample_weight=sample_weights,
            callbacks=model_callbacks
        )

        # 保存最终模型
        self.model.save(f"trained_models/fold_{self.fold_index + 1}_final_model.h5")
        logger.info(f"第 {self.fold_index + 1} 折模型已保存")
        return history

    def _get_samples_weights(self, y):
        """计算样本权重（解决类别不平衡）"""
        class_counts = np.bincount(y.flatten().astype(int))
        total = len(y.flatten())
        weights = total / (len(class_counts) * class_counts)
        sample_weights = np.array([weights[int(label)] for label in y.flatten()])
        return sample_weights.reshape(y.shape)

    def predict(self, X):
        """预测函数（优化输入处理）"""
        X_pad = []
        for item in X:
            if isinstance(item, list):
                try:
                    x_arr = np.array(item, dtype='float32')
                except ValueError:
                    processed = []
                    for elem in item:
                        elem = np.array(elem, dtype='float32') if isinstance(elem, list) else elem
                        processed.append(elem)
                    x_arr = np.array(processed, dtype='float32')
            elif isinstance(item, np.ndarray):
                x_arr = item.astype('float32')
            else:
                raise ValueError(f"不支持的预测输入类型: {type(item)}")

            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(-1, 1)
            
            # 统一预测输入形状
            if x_arr.shape[0] < self.max_seq_length:
                pad_length = self.max_seq_length - x_arr.shape[0]
                x_padded = np.pad(
                    x_arr, 
                    pad_width=((0, pad_length), (0, 0)),
                    mode='constant', 
                    constant_values=-100.0
                )
            else:
                x_padded = x_arr[:self.max_seq_length, :]

            if x_padded.shape[1] < self.feature_dim:
                pad_feat = self.feature_dim - x_padded.shape[1]
                x_padded = np.pad(
                    x_padded,
                    pad_width=((0, 0), (0, pad_feat)),
                    mode='constant',
                    constant_values=-100.0
                )
            else:
                x_padded = x_padded[:, :self.feature_dim]

            X_pad.append(x_padded.reshape(1, self.max_seq_length, self.feature_dim, 1))

        X_processed = np.concatenate(X_pad, axis=0)
        return self.model.predict(X_processed)

    def clear_params(self):
        """清除模型参数（用于交叉验证重置）"""
        self.model = self._build_model()
        self.classes_ = None
        self.max_seq_length = None


# 继承基础类实现完整DeepSound模型
class DeepSound(DeepSoundBaseRNN):
    def __init__(self, **kwargs):
        super().__init__(** kwargs)

    def _build_model(self):
        """重写模型构建方法，使用更复杂的架构"""
        model = Sequential([
            layers.Input(shape=(self.max_seq_length, self.feature_dim, 1)),
            
            # 卷积层块
            layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),
            
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),
            
            # 时序特征提取
            layers.TimeDistributed(layers.Flatten()),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.BatchNormalization(),
            
            # 输出层
            layers.TimeDistributed(layers.Dense(self.output_size, activation='softmax'))
        ])

        model.compile(
            optimizer=Adagrad(learning_rate=0.005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.SparseCategoricalCrossentropy()]
        )
        return model