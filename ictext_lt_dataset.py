import os
from torch.utils.data import Dataset
from extra_utils import get_char2idx_dict
from PIL import Image
import numpy as np

class ICTextLTDataset(Dataset):
    def __init__(self, data_dir, split, imb_factor, imb_num_class=36, img_max=3200, imb_type='exp', transform=None):
        self.image_path = os.path.join(data_dir, split, 'imgs')
        self.label_path = os.path.join(data_dir, split, f'{split}_with_rot.txt')
        self.imb_num_class = imb_num_class
        self.img_max = img_max
        self.imb_factor = imb_factor

        self.class_dict = get_char2idx_dict()
        self.all_paths = []
        self.class_freq = [0] * 62
        self.targets = []

        with open(self.label_path, mode='r') as in_txt:
            lines = in_txt.readlines()
            for line in lines:
                line = line.strip().split(' ')
                self.targets.append(int(self.class_dict[line[1]]))
                self.class_freq[int(self.class_dict[line[1]])] += 1
                self.all_paths.append(line[0])
        
        if split == 'train':
            if imb_factor == -1: # original imbalance, 6155/334 = 18.43
                img_num_list = self.class_freq[:self.imb_num_class]
            else:
                img_num_list = self.get_img_num_per_cls(self.imb_num_class, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
        else:
            class_freq = np.asarray(self.class_freq[:self.imb_num_class])
            min_num_image = class_freq.min()
            img_num_list = [min_num_image] * self.imb_num_class
            self.gen_imbalanced_data(img_num_list)
        
        self.class_freq = img_num_list
        self.transform = transform
        
        assert len(self.all_paths) == len(self.targets)
        
        print("{} Mode: Contain {} images".format(split, len(self.targets)))
    
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = self.img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(self.img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(self.img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(self.img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        classes = classes[:self.imb_num_class]

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.extend([self.all_paths[i] for i in selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
            
        self.all_paths = new_data
        self.targets = new_targets

    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        current_image, label = self.all_paths[idx], self.targets[idx]
        image = Image.open(os.path.join(self.image_path, current_image))

        if not isinstance(self.transform, str):
            image = self.transform(image)
 
        return image, (label)

    def get_cls_num_list(self, verbose=False):
        return self.class_freq
        
if __name__ == '__main__':
    ictext_LT_train = ICTextLTDataset('/home/jun/Documents/GitHub/ols/data/ictext2021', 'train', imb_factor=0.01,
                                      imb_num_class=36, img_max=3200,  imb_type='exp', transform=None)
    ictext_LT_test = ICTextLTDataset('/home/jun/Documents/GitHub/ols/data/ictext2021', 'val', imb_factor=0.01,
                                     imb_num_class=36, img_max=3200, imb_type='exp', transform=None)
    
    assert np.asarray(ictext_LT_train.class_freq).sum() == 25711
    assert np.all(np.asarray(ictext_LT_test.class_freq) == 175)
    assert np.all(np.asarray(ictext_LT_train.targets) < 36)
    assert np.all(np.asarray(ictext_LT_test.targets) < 36)