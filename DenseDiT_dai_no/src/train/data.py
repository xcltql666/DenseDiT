from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import os

# from DenseDiT.DenseDiT_dai_no.inference import context_path
    
class DenseDiTDataset(Dataset):
    def __init__(
        self,
        image_dir,
        condition_dir,
        context_file,
        descriptions,
        resize=(512, 512)
    ):
        self.image_dir = image_dir
        self.condition_dir = condition_dir
        self.context_file = context_file
        self.descriptions = descriptions
        self.resize = resize
        self.file_names = list(descriptions.keys())

        self.to_tensor = T.ToTensor()
    
    def load_images(self, image_dir, condition_dir, file_name, context_file):
        context_image = os.path.join(context_file, f"{file_name}.jpg")
        
        # print(condition_dir)
        base = ""
        if file_name[-4:] == "left":
            base = f"{file_name[:-5]}_right"
        else:
            base = f"{file_name[:-5]}left"
        image_path = os.path.join(image_dir, f"{base}.jpg")
        condition_path = os.path.join(condition_dir, f"{base}_pf.jpg")
        # print(image_path, condition_path, context_image, "111111111")
        image = Image.open(image_path).convert("RGB").resize(self.resize)
        condition_image = Image.open(condition_path).convert("RGB").resize(self.resize)
        context_image = Image.open(context_image).convert("RGB").resize(self.resize)

        return image, condition_image, context_image

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        description = self.descriptions[file_name]

        image, condition_img, context_image = self.load_images(self.image_dir, self.condition_dir, file_name, self.context_file)

        return {
            "image": self.to_tensor(image),
            "condition": self.to_tensor(condition_img),
            "context": self.to_tensor(context_image),
            "description": description,
        }
