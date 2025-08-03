from collections import namedtuple
import os
import glob

# 动态计算项目根目录，避免硬编码绝对路径
# 当前文件路径：DeepSound/src/data/utils_data_sources.py
# 向上三级目录即为DeepSound根目录
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__), "../../..")
DATA_SOURCES_PATH = os.path.join(PROJECT_ROOT, "data")  # 指向DeepSound/data目录


def list_datasets():
    Dataset = namedtuple("dataset",
                         ["str_id", "name", "folder", "audio_files_format",
                          "audio_sampling_frequency", "imu_sampling_frequency", "multimodal"])

    # 验证路径是否存在
    if not os.path.exists(DATA_SOURCES_PATH):
        # 打印详细错误信息
        print(f"致命错误：数据根目录不存在！")
        print(f"当前设置的路径：{DATA_SOURCES_PATH}")
        print(f"请检查路径是否正确，或修改 DATA_SOURCES_PATH 变量的值")
        exit(1)

    datasets = {
        'zavalla2022': Dataset(
            str_id="jm2022",
            name="Jaw Movement 2022 Dataset",
            folder=os.path.join(DATA_SOURCES_PATH, "raw", "jm2022"),  # 补充raw层级
            audio_files_format="wav",
            audio_sampling_frequency=22050,
            imu_sampling_frequency=None,
            multimodal=False
        ),
        'jaw_movements2020': Dataset(
            str_id="jm2020",
            name="Jaw Movement 2020 Dataset",
            folder=os.path.join(DATA_SOURCES_PATH, "raw", "jm2020"),  # 补充raw层级，指向data/raw/jm2020
            audio_files_format="wav",
            audio_sampling_frequency=22050,
            imu_sampling_frequency=None,
            multimodal=False
        )
    }
    
    print(f"已确认数据根目录存在：{DATA_SOURCES_PATH}")
    print(f"jm2020数据集路径：{datasets['jaw_movements2020'].folder}")
    return datasets


def get_files_in_dataset(dataset):
    """ 适配你的文件结构：
        - 标签文件在 jm2020/labels/recording_01.txt
        - 音频文件在 jm2020/audios/recording_01.wav
    """
    if dataset.audio_files_format == "wav":
        ext = "wav"

    dataset_files = []
    # 1. 只查找 labels 子文件夹里的 .txt 文件（非递归，因为结构固定）
    labels_dir = os.path.join(dataset.folder, "labels")
    labels_file_list = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))

    for label_file in labels_file_list:
        # 2. 提取文件名（如 "recording_01.txt" → "recording_01"）
        base_name = os.path.splitext(os.path.basename(label_file))[0]  
        # 3. 构建对应的音频文件路径（audios/recording_01.wav）
        audio_file = os.path.join(dataset.folder, "audios", f"{base_name}.{ext}")  

        # 4. 检查文件是否存在（若不存在，直接报错终止）
        if not os.path.isfile(audio_file):
            print(f"\n错误：音频文件不存在！\n标签文件：{label_file}\n期望音频：{audio_file}")
            exit(1)
        if not os.path.isfile(label_file):
            print(f"\n错误：标签文件不存在！\n路径：{label_file}")
            exit(1)

        # 5. 配对成功，加入列表
        dataset_files.append( (audio_file, label_file) )

    return dataset_files


if __name__ == "__main__":
    # 强制检查路径是否正确
    print(f"正在验证路径：{DATA_SOURCES_PATH}")
    if not os.path.exists(DATA_SOURCES_PATH):
        print(f"路径不存在！请修改 DATA_SOURCES_PATH 为正确的绝对路径")
        exit(1)
    
    datasets = list_datasets()
    if 'jaw_movements2020' in datasets:
        jm2020_dataset = datasets['jaw_movements2020']
        if not os.path.exists(jm2020_dataset.folder):
            print(f"jm2020数据集文件夹不存在：{jm2020_dataset.folder}")
            exit(1)
            
        files = get_files_in_dataset(jm2020_dataset)
        print(f"成功找到 {len(files)} 组文件")