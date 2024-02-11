from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os,json
import cv2
from scipy.io import loadmat


def image_id_to_path(image_id):
    e_num = image_id[:2]
    s_num = image_id[2:4]
    a_num = image_id[4:6]
    frame_num = image_id[6:]

    img_path = f"E{int(e_num):02}/S{int(s_num):02}/A{int(a_num):02}/rgb/frame{int(frame_num):03}.png"
    csi_path=f"E{int(e_num):02}/S{int(s_num):02}/A{int(a_num):02}/wifi-csi/frame{int(frame_num):03}.mat"
    return img_path,csi_path


class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations, transform=None):
        with open(annotations, 'r') as f:
            self.annotations = json.load(f)

        self.images_dir = images_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_id = self.annotations[idx]['image_id']
        img_path,csi_path = image_id_to_path(image_id)

        img_path = os.path.join(self.images_dir, img_path)
        csi_path = os.path.join(self.images_dir, csi_path)

        csi = loadmat(csi_path)
        CSI={'phase':csi['CSIphase'],'amp':csi['CSIamp']}
        
      
        image = cv2.imread(img_path)
        Image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        
      
        label = self.annotations[idx]

        data = {"image":Image,"csi":CSI}

        return data, label


# images_dir = "/home/visier/mm_fi/MMFi_dataset/all_images"
# annotations ="/home/visier/mm_fi/MMFi_dataset/all_images/kp_dump_results/all_annotations.json"

# # 创建自定义数据集和数据加载器
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
