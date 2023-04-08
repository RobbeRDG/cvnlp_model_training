import torch
from torch.utils.data import Dataset

class MELDDataset(Dataset):
    def __init__(
        self,
        labels,
        audio_folder_path,
        transform=None,
        target_transform=None
    ):
        self.labels = labels
        self.audio_folder_path = audio_folder_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        # Get the label
        label = self.labels.iloc[index]

        # Get the audio file path
        audio_file_name = label['Utterance'].replace('.mp4', '.wav')
        audio_file_path = join(self.audio_folder_path, audio_file_name)

        # Load the audio
        audio = load_audio(audio_file_path)

        # Transform the audio
        if self.transform:
            audio = self.transform(audio)

        # Transform the label
        if self.target_transform:
            label = self.target_transform(label)

        return audio, label