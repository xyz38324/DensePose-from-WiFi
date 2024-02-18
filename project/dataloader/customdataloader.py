from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os,json,cv2,torch
import copy
import pycocotools.mask as mask_util
import numpy as np
from scipy.io import loadmat
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.layers import ROIAlign
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from typing import Any, Dict, List, Tuple
from densepose.structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData

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
        dataset_dict = self.annotations[idx]
        dataset_dict = self.transform(dataset_dict)

        img_path,csi_path = image_id_to_path(image_id)

        img_path = os.path.join(self.images_dir, img_path)
        csi_path = os.path.join(self.images_dir, csi_path)

        csi = loadmat(csi_path)
        CSI_phase = torch.from_numpy(csi['CSIphase']).float()
        CSI_amp = torch.from_numpy(csi['CSIamp']).float() 
        CSI={'phase':CSI_phase,'amp':CSI_amp}
        
      
        dataset_dict['csi']=CSI
        

   

        return dataset_dict

def custom_collate_fn(batch):
    dict_list = []

    for data in batch:
        dict_list.append(data)

    return dict_list


class DatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, augmentation,is_train=True):
        self.augmentation = augmentation

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = (
            cfg.MODEL.MASK_ON or (
                cfg.MODEL.DENSEPOSE_ON
                and cfg.MODEL.ROI_DENSEPOSE_HEAD.COARSE_SEGM_TRAINED_BY_MASKS)
        )
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.densepose_on   = cfg.MODEL.DENSEPOSE_ON
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = None#utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        # else:
        #     self.keypoint_hflip_indices = None

        if self.densepose_on:
            densepose_transform_srcs = [
                MetadataCatalog.get(ds).densepose_transform_src
                for ds in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
            ]
         
            densepose_transform_data_fpath = PathManager.get_local_path(densepose_transform_srcs[0])
            self.densepose_transform_data = DensePoseTransformData.load(
                densepose_transform_data_fpath
            )

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  
        img_dir = "/home/visier/mm_fi/MMFi_dataset/all_images"
        img_file=dataset_dict["file_name"]
        img_path = os.path.join(img_dir, img_file)
        
        image = utils.read_image(img_path, format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.augmentation, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        for anno in dataset_dict["annotations"]:
            if not self.mask_on:
                anno.pop("segmentation", None)
            # if not self.keypoint_on:
            #     anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        # USER: Don't call transpose_densepose if you don't need
        annos = [
            self._transform_densepose(
                transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                ),
                transforms,
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

       

        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        densepose_annotations = [obj.get("densepose") for obj in annos]
        if densepose_annotations and not all(v is None for v in densepose_annotations):
            instances.gt_densepose = DensePoseList(
                densepose_annotations, instances.gt_boxes, image_shape
            )

        dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict

    def _transform_densepose(self, annotation, transforms):
        if not self.densepose_on:
            return annotation

        # Handle densepose annotations
        is_valid, reason_not_valid = DensePoseDataRelative.validate_annotation(annotation)
        if is_valid:
            densepose_data = DensePoseDataRelative(annotation, cleanup=True)
            densepose_data.apply_transform(transforms, self.densepose_transform_data)
            annotation["densepose"] = densepose_data
        else:
            # logger = logging.getLogger(__name__)
            # logger.debug("Could not load DensePose annotation: {}".format(reason_not_valid))
            DensePoseDataRelative.cleanup_annotation(annotation)
            # NOTE: annotations for certain instances may be unavailable.
            # 'None' is accepted by the DensePostList data structure.
            annotation["densepose"] = None
        return annotation

    def _add_densepose_masks_as_segmentation(
        self, annotations: List[Dict[str, Any]], image_shape_hw: Tuple[int, int]
    ):
        for obj in annotations:
            if ("densepose" not in obj) or ("segmentation" in obj):
                continue
            # DP segmentation: torch.Tensor [S, S] of float32, S=256
            segm_dp = torch.zeros_like(obj["densepose"].segm)
            segm_dp[obj["densepose"].segm > 0] = 1
            segm_h, segm_w = segm_dp.shape
            bbox_segm_dp = torch.tensor((0, 0, segm_h - 1, segm_w - 1), dtype=torch.float32)
            # image bbox
            x0, y0, x1, y1 = (
                v.item() for v in BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            )
            segm_aligned = (
                ROIAlign((y1 - y0, x1 - x0), 1.0, 0, aligned=True)
                .forward(segm_dp.view(1, 1, *segm_dp.shape), bbox_segm_dp)
                .squeeze()
            )
            image_mask = torch.zeros(*image_shape_hw, dtype=torch.float32)
            image_mask[y0:y1, x0:x1] = segm_aligned
            # segmentation for BitMask: np.array [H, W] of bool
            obj["segmentation"] = image_mask >= 0.5


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        kp  = annotation["keypoints"]
        kp_array = np.array(kp, dtype=np.float32).reshape(-1, 3)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # kp_tensor = torch.tensor(kp_array)#, device=device)

       

        annotation["keypoints"]=kp_array
    return annotation