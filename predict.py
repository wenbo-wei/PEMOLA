# Copyright (c) 2026 Wenbo Wei.
# Licensed under the MIT License (see LICENSE).
# PEMOLA: Occlusion-Aware Panoptic Segmentation with Joint Position Embedding
# and Occlusion-Level Attention (ICME 2026).

import os
import cv2
import json
import torch
from PIL import Image

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

# import Mask2Former project
from mask2former import add_maskformer2_config
from mask2former import add_pemola_config

class Predictor:
    def setup(self):
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_pemola_config(cfg)
        cfg.merge_from_file("configs/coco_olac/panoptic-segmentation/pemola_R50_bs16_50ep.yaml")
        # cfg.MODEL.WEIGHTS = './model_zoo/coco_r50.pkl'
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

        # cfg.MODEL.WEIGHTS = "output/model_final_a100base.pth"
        # cfg.MODEL.PEMOLA.PE_MODULATION = False
        cfg.MODEL.WEIGHTS = "output/model_final_pemola.pth"
        cfg.MODEL.PEMOLA.PE_MODULATION = True

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

        if self.cfg.MODEL.PEMOLA.PE_MODULATION:
            self.dataset_root = os.getenv("DETECTRON2_DATASETS", "datasets")
            self.occlusion_maps = {'low': 0, 'mid': 1, 'high': 2}

            if self.cfg.INPUT.DATASET_MAPPER_NAME == "coco_olac_panoptic_lsj":
                self.cam_dir = os.path.join(self.dataset_root, "coco_olac_cam/cam_pt_val")
                occlusion_label_json = os.path.join(self.dataset_root, "coco_olac/val/occlusion_label_val.json")
            elif self.cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
                self.cam_dir = os.path.join(self.dataset_root, "cityscapes_cam/cam_pt_val")
                occlusion_label_json = os.path.join(self.dataset_root, "cityscapes/gtFine/occlusion_label_val.json")
            else:
                ValueError(f"Unsupported cfg.INPUT.DATASET_MAPPER_NAME: {self.cfg.INPUT.DATASET_MAPPER_NAME}.")

            with open(occlusion_label_json, 'r') as olj:
                self.occlusion_label_ann = json.load(olj)

    def predict(self, image):
        auxilary = {}
        img_name = os.path.splitext(os.path.basename(image))[0]
        # img_id = img_name.lstrip("0")
        auxilary["occlusion_label"] = self.occlusion_maps[self.occlusion_label_ann[img_name]]
        cam = torch.load(os.path.join(self.cam_dir, f"{img_name}.pt"), weights_only=False).numpy()
        # cam_resized = cv2.resize(cam, (auxilary['width'], auxilary['height']),
        #                          interpolation=cv2.INTER_LINEAR)
        # cam_transformed = transforms.apply_image(cam_resized)
        # auxilary["occlusion_cam"] = torch.as_tensor(np.ascontiguousarray(cam_transformed))
        auxilary["occlusion_cam"] = cam
        im = cv2.imread(str(image))
        outputs = self.predictor(im, auxilary)
        v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"),
                                              outputs["panoptic_seg"][1]).get_image()
        # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()
        # v = Visualizer(im[:, :, ::-1], self.coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
        # semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        # result = np.concatenate((panoptic_result, instance_result, semantic_result), axis=0)[:, :, ::-1]

        # model_tag = "pemola" if self.cfg.MODEL.PEMOLA.PE_MODULATION else "base"
        # out_name = f"{img_name}_{model_tag}.pdf"
        out_name = f"{img_name}.pdf"
        out_path = os.path.join(".", out_name)
        img = Image.fromarray(panoptic_result)
        img.save(out_path, "PDF")
        return out_path


if __name__ == "__main__":
    image_predictor = Predictor()
    im_dir = '/home/wenbo/data/datasets/coco_olac/val/val/000000055167.jpg'
    image_predictor.setup()
    image_predictor.predict(im_dir)