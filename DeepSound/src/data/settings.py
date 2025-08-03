import os

# 动态计算项目根目录（当前文件路径：DeepSound/src/data/settings.py）
# 向上三级目录即为DeepSound根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# 数据根目录统一为项目根目录下的data文件夹
DATA_SOURCES_PATH = os.path.join(PROJECT_ROOT, "data")

# 缓存目录规范到data/interim/cache（与utils_data_sources.py保持路径一致性）
CACHE_DIR = os.path.join(DATA_SOURCES_PATH, "interim", "cache")

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)