import os
import pickle
import numpy as np
import librosa
from transformers import Wav2Vec2Processor
from base import Datum, DatasetBase, DATASET_REGISTRY, DataManager

@DATASET_REGISTRY.register()
class Vocaset(DatasetBase):

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.ROOT))
        self.dataset_dir = os.path.join(root, cfg.NAME)
        self.split_path = os.path.join(self.dataset_dir, cfg.SPLIT)
        self.audio_path = os.path.join(self.dataset_dir, cfg.VOCASET.AUDIO)
        self.vertices_path = os.path.join(self.dataset_dir, cfg.VOCASET.VERTICES)

        template_file = os.path.join(self.dataset_dir, cfg.VOCASET.TEMPLATE)
        with open(template_file, 'rb') as fin:
            templates = pickle.load(fin, encoding='latin1')

        if cfg.VOCASET.READ_AUDIO: # read_audio==False when training vq to save time
            processor = Wav2Vec2Processor.from_pretrained(cfg.VOCASET.WAV2VEC2)

        subjects_dict = {"train": [], "val": [], "test": []}
        with open(self.split_path) as f:
            for line in f:
                split, name = line.strip().split()
                subjects_dict[split].append(name)

        train, val, test = [], [], []
        # Walk through the audio directory and process each WAV file
        for file in os.listdir(self.audio_path):
            # Check if the file is a WAV file
            if file.endswith("wav"):
                # If the configuration allows reading audio, load the audio file
                if cfg.VOCASET.READ_AUDIO:
                    wav_path = os.path.join(self.audio_path, file)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    audio = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                else:
                    audio = None
                # Create a key for the data dictionary using the file name
                key = file.replace("wav", "npy")
                subject_id = "_".join(key.split("_")[:-1])
                template = templates[subject_id].reshape((-1))

                # Construct the path to the vertex file
                vertices_path = os.path.join(self.vertices_path, key)

                if os.path.exists(vertices_path):
                    vertices = np.load(vertices_path, allow_pickle=True)[::2, :]  # due to the memory limit

                    # sum the dataset
                    data = Datum(file, audio, vertices, template)
                    sentence_id = int(key.split(".")[0][-2:])
                    if subject_id in subjects_dict["train"] and sentence_id in cfg.VOCASET.TRAIN:
                        train.append(data)
                    if subject_id in subjects_dict["val"] and sentence_id in cfg.VOCASET.VAL:
                        val.append(data)
                    if subject_id in subjects_dict["test"] and sentence_id in cfg.VOCASET.TEST:
                        test.append(data)

        super().__init__(train=train, val=val, test=test)


class CodeTalkerDataManager(DataManager):
  
    def __init__(self,
                assistant,
                dataset_wrapper=None):
        super().__init__(assistant, dataset_wrapper)