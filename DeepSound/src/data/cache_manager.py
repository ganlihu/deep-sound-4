import os
import logging
import joblib
import hashlib
from datetime import datetime as dt

import pandas as pd

# 修正导入路径，使用当前项目的settings模块
from .settings import CACHE_DIR  # 从同目录下的settings导入CACHE_DIR

logger = logging.getLogger(__name__)


CACHE_INDEX_NAME = 'cache_index.txt'


class DatasetCache():
    ''' Implement cache-like structure to store and retrieve datasets files. '''

    def __init__(self):
        self.timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

        # 使用settings中定义的缓存目录
        self.index_file = os.path.join(CACHE_DIR, CACHE_INDEX_NAME)

        # 确保缓存目录存在
        os.makedirs(CACHE_DIR, exist_ok=True)

        if not os.path.isfile(self.index_file):
            self.cache_index = pd.DataFrame(columns=['item', 'params'])
        else:
            # 读取已有的缓存索引文件（处理可能的编码问题）
            self.cache_index = pd.read_csv(
                self.index_file, 
                index_col=0, 
                names=['item', 'params'], 
                sep='\t',
                encoding='utf-8'
            )

    def __get_filters_names__(self, **kargs):
        filters = []
        if ('filters' in kargs) and kargs['filters']:
            for i in kargs['filters']:
                filters.append((i[0].__name__, i[1]))

        kargs['filters'] = filters

        return kargs

    def load(self,** kargs):
        kargs = self.__get_filters_names__(**kargs)
        cache_item_key = '__'.join([f'{str(key)}-{str(value)}' for key, value in kargs.items()])
        logger.info(f"尝试缓存键: {cache_item_key}")

        item_key = hashlib.sha256(cache_item_key.encode(encoding='UTF-8')).hexdigest()

        if item_key in self.cache_index.index:
            cache_item_match = self.cache_index.loc[item_key]
            # 验证缓存项唯一性
            assert sum([m == item_key for m in self.cache_index.index]) == 1, \
                f'缓存缓存项存在重复条目! {cache_item_key}'

            # 从统一缓存缓存目录加载数据
            cache_item_path = os.path.join(CACHE_DIR, cache_item_match['item'])
            if not os.path.exists(cache_item_path):
                logger.warning(f"缓存文件不存在，将重新生成: {cache_item_path}")
                return None

            cache_item = joblib.load(cache_item_path)
            X = cache_item['X']
            y = cache_item['y']

            return (X, y)

        return None

    def save(self, X, y, **kargs):
        kargs = self.__get_filters_names__(** kargs)
        cache_item_key = '__'.join([f'{str(key)}-{str(value)}' for key, value in kargs.items()])

        cache_item = {
            'X': X,
            'y': y
        }

        # 保存缓存文件到统一的缓存目录
        cache_item_path = os.path.join(CACHE_DIR, self.timestamp + '.pkl')
        joblib.dump(cache_item, cache_item_path)

        # 更新缓存索引
        item_key = hashlib.sha256(cache_item_key.encode(encoding='UTF-8')).hexdigest()
        self.cache_index.loc[item_key] = [self.timestamp + '.pkl', cache_item_key]
        # 保存索引文件（指定编码为utf-8，避免跨平台问题）
        self.cache_index.to_csv(self.index_file, header=None, sep='\t', encoding='utf-8')
        logger.info(f"缓存已保存: {cache_item_path}")