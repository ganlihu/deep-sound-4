from chewbite_fusion.features.base import BaseFeatureBuilder
from chewbite_fusion.features import audio_raw_data as ard


class BaseFeatureFactoryNoPreprocessing():
    def __init__(self, features,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        self.features = []
        for feature in features:
            self.features.append(BaseFeatureBuilder(
                feature,
                audio_sampling_frequency,
                movement_sampling_frequency,
                None
            ))


class FeatureFactory_RawAudioData(BaseFeatureFactoryNoPreprocessing):
    def __init__(self,
                 audio_sampling_frequency,
                 movement_sampling_frequency):
        features = [
            ard.AudioRawData
        ]
        super().__init__(features,
                         audio_sampling_frequency,
                         movement_sampling_frequency)