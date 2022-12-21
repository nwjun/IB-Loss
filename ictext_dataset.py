import os
from torch.utils.data import Dataset
from extra_utils import get_char2idx_dict
from PIL import Image

class ICTextDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.image_path = os.path.join(data_dir, split, 'imgs')
        self.label_path = os.path.join(data_dir, split, f'{split}_with_rot.txt')
        
        self.gt_dict = {}

        self.class_dict = get_char2idx_dict()
        self.all_paths = []
        self.class_freq = [0] * 66
        self.targets = []

        with open(self.label_path, mode='r') as in_txt:
            lines = in_txt.readlines()
            for line in lines:
                line = line.strip().split(' ')

                self.gt_dict[line[0]] = int(self.class_dict[line[1]])
                self.targets.append(int(self.class_dict[line[1]]))
                self.class_freq[int(self.class_dict[line[1]])] += 1
                self.all_paths.append(line[0])
                    
        self.transform = transform

    def __len__(self):
        return len(self.gt_dict)

    def __getitem__(self, idx):
        current_image = self.all_paths[idx]
        image = Image.open(os.path.join(self.image_path, current_image))

        if not isinstance(self.transform, str):
            image = self.transform(image)

        label = self.gt_dict[os.path.basename(current_image)]
        return image, label

    def get_cls_num_list(self, verbose=False):
        return self.class_freq