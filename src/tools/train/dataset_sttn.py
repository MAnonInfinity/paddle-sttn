import os
import json
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.tools.train.utils_sttn import ZipReader, create_random_shape_with_random_motion
from src.tools.train.utils_sttn import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip


# Custom dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        # Initialization function, takes config dict and split type (default 'train')
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']  # Sample length parameter
        self.size = self.w, self.h = (args['w'], args['h'])  # Set target width and height

        # Open json file containing data info
        with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
            self.video_dict = json.load(f)  # Load json content
        self.video_names = list(self.video_dict.keys())  # Get list of video names
        if debug or split != 'train':  # If debug mode or not training set, take only first 100 videos
            self.video_names = self.video_names[:100]

        # Define data transformations to stacked tensors
        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),  # Tensor format for PyTorch
        ])

    def __len__(self):
        # Return number of videos in dataset
        return len(self.video_names)

    def __getitem__(self, index):
        # Get a sample item
        try:
            item = self.load_item(index)  # Try to load data item at specified index
        except:
            print('Loading error in video {}'.format(self.video_names[index]))  # If loading error, print error message
            item = self.load_item(0)  # Load first item as fallback
        return item

    def load_item(self, index):
        # Specific implementation of loading data item
        video_name = self.video_names[index]  # Get video name based on index
        # Generate list of frame filenames for all video frames
        all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(self.video_dict[video_name])]
        # Generate mask with random shape and random motion
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
        # Get reference frame indices
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # Read video frames
        frames = []
        masks = []
        for idx in ref_index:
            # Read image, convert to RGB, resize and add to list
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name), all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            # If training set, random horizontal flip
            frames = GroupRandomHorizontalFlip()(frames)
        # Convert to tensor format
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0  # Normalization
        mask_tensors = self._to_tensors(masks)  # Convert mask to tensor
        return frame_tensors, mask_tensors  # Return image and mask tensors


def get_ref_index(length, sample_length):
    # Implementation of getting reference index
    if random.uniform(0, 1) > 0.5:
        # 50% chance to randomly sample frames
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()  # Sort to ensure order
    else:
        # 50% chance to select continuous frames
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
