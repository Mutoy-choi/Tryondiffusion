import numpy as np
from PIL import Image
from scipy.ndimage import sobel
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import openpose as op
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from carvekit.api.high import HiInterface

class TryOnDiffusionDataset(Dataset):
    def __init__(self, image_paths, densepose_config_path, densepose_weights_path, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(op.WrapperPython.get_default_wrapper_options())
        self.opWrapper.start()
        self.densepose_predictor = self._init_densepose_predictor(densepose_config_path, densepose_weights_path)
        self.bg_remover = HiInterface(object_type="object")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load person image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        # Predict human parsing map and 2D pose keypoints for person image using DensePose
        Sp, Jp = self.get_densepose_output(image)

        # Predict human parsing map and 2D pose keypoints for person image using OpenPose
        Sp_op, Jp_op = self.get_openpose_output(image)

        # Load garment image (assuming you have a separate list of garment image paths)
        garment_image_path = self.garment_image_paths[idx]
        garment_image = Image.open(garment_image_path)

        # Predict human parsing map and 2D pose keypoints for garment image using OpenPose
        _, Jg = self.get_openpose_output(garment_image)

        # Segment out the garment from the garment image
        Ic = self.segment_clothing(garment_image, Sp)

        # Generate clothing-agnostic RGB image for the person image
        Ia = self.generate_clothing_agnostic_image(image, Sp, Jp)

        if self.transform:
            Ia = self.transform(Ia)
            Ic = self.transform(Ic)

        return Ia, Jp, Ic, Jg

    def get_openpose_output(self, image):
        # 이미지를 numpy 배열로 변환
        image_np = np.array(image)

        # OpenPose를 사용하여 포즈와 키포인트 예측
        datum = op.Datum()
        datum.cvInputData = image_np
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # 파싱 맵 (Sp)와 키포인트 (Jp) 반환
        Sp = datum.poseKeypoints  # 파싱 맵
        Jp = datum.poseHeatMaps  # 키포인트
        # 키포인트 정규화
        Jp = Jp / Jp.max()
        return Sp, Jp

    def reverse_label(img_path, output_path):
        im = Image.open(img_path)
        im_np = np.array(im)

        # 이미지를 좌우 반전
        rev_im = np.fliplr(im_np)

        # 라벨 값을 교환
        label_pairs = [(14, 15), (16, 17), (18, 19)]
        for label1, label2 in label_pairs:
            mask1 = (rev_im == label1)
            mask2 = (rev_im == label2)
            rev_im[mask1] = label2
            rev_im[mask2] = label1

        # 결과 이미지 저장
        Image.fromarray(rev_im).save(output_path)

    def write_edge(img_path, output_path, vis_output_path):
        im = Image.open(img_path)
        im_np = np.array(im)

        # 그래디언트를 계산하여 엣지 정보 추출
        dx = sobel(im_np, axis=0, mode='constant')
        dy = sobel(im_np, axis=1, mode='constant')
        magnitude = np.hypot(dx, dy)
        instance_contour = (magnitude > 0).astype(np.uint8)

        # 결과 이미지 저장
        Image.fromarray(instance_contour).save(output_path)
        Image.fromarray(instance_contour * 255).save(vis_output_path)

    def _init_densepose_predictor(self, config_path, weights_path):
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = weights_path
        return DefaultPredictor(cfg)

    def get_densepose_output(self, image):
        # 이미지를 OpenCV 형식으로 변환
        image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # DensePose 예측 수행
        outputs = self.densepose_predictor(image_cv2)
        # 파싱 맵 추출
        densepose = outputs["instances"].pred_densepose
        # S는 파싱 맵을 나타내는 텐서입니다.
        parsing_map = densepose.S.cpu().numpy()[0]
        return parsing_map

    def segment_clothing(self, image, parsing_map):
        # 파싱 맵에서 의복 영역을 나타내는 라벨 값 (예: 2)을 사용하여 마스크 생성
        clothing_mask = (parsing_map == 2).astype(np.uint8)
        image_np = np.array(image)
        Ic = image_np * np.repeat(clothing_mask[:, :, np.newaxis], 3, axis=2)
        return Image.fromarray(Ic)

    def generate_clothing_agnostic_image(self, image, parsing_map, keypoints):
        # 파싱 맵에서 의복을 제외한 나머지 영역을 사용하여 마스크 생성
        non_clothing_mask = (parsing_map != 2).astype(np.uint8)
        image_np = np.array(image)
        Ia = image_np * np.repeat(non_clothing_mask[:, :, np.newaxis], 3, axis=2)
        return Image.fromarray(Ia)
