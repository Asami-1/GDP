from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmdet.apis import inference_detector, init_detector



# optional
return_heatmap = False
# e.g. use ('backbone', ) to return backbone feature
output_layer_names = None



# # Initialize detection model
det_config=r"ViTPose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
det_checkpoint=r"https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

det_model = init_detector(det_config, det_checkpoint, device='cuda:0')

mmdet_results = inference_detector(det_model, "./test.jpg")

person_results = process_mmdet_results(mmdet_results, 1)


pose_model = init_pose_model("./configs/ViTPose_huge_simple_coco_256x192.py","./vitpose-h-simple.pth")


pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            "./test.jpg",
            person_results,
            bbox_thr=0.3,
            format='xyxy')

vis_pose_result(
            pose_model,
            "./test.jpg",
            pose_results,
            out_file='./test_labeled.jpg')



# class PoseModel:
#     MODEL_DICT = {
#         'ViTPose-B (single-task train)': {
#             'config':
#             'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
#             'model': 'models/vitpose-b.pth',
#         },
#         'ViTPose-L (single-task train)': {
#             'config':
#             'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
#             'model': 'models/vitpose-l.pth',
#         },
#         'ViTPose-B (multi-task train, COCO)': {
#             'config':
#             'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
#             'model': 'models/vitpose-b-multi-coco.pth',
#         },
#         'ViTPose-L (multi-task train, COCO)': {
#             'config':
#             'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
#             'model': 'models/vitpose-l-multi-coco.pth',
#         },
#     }

#     def __init__(self, device: str | torch.device):
#         self.device = torch.device(device)
#         self.model_name = 'ViTPose-B (multi-task train, COCO)'
#         self.model = self._load_model(self.model_name)

#     def _load_all_models_once(self) -> None:
#         for name in self.MODEL_DICT:
#             self._load_model(name)

#     def _load_model(self, name: str) -> nn.Module:
#         dic = self.MODEL_DICT[name]
#         ckpt_path = huggingface_hub.hf_hub_download('hysts/ViTPose',
#                                                     dic['model'],
#                                                     use_auth_token=HF_TOKEN)
#         model = init_pose_model(dic['config'], ckpt_path, device=self.device)
#         return model

#     def set_model(self, name: str) -> None:
#         if name == self.model_name:
#             return
#         self.model_name = name
#         self.model = self._load_model(name)

#     def predict_pose_and_visualize(
#         self,
#         image: np.ndarray,
#         det_results: list[np.ndarray],
#         box_score_threshold: float,
#         kpt_score_threshold: float,
#         vis_dot_radius: int,
#         vis_line_thickness: int,
#     ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
#         out = self.predict_pose(image, det_results, box_score_threshold)
#         vis = self.visualize_pose_results(image, out, kpt_score_threshold,
#                                           vis_dot_radius, vis_line_thickness)
#         return out, vis

#     def predict_pose(
#             self,
#             image: np.ndarray,
#             det_results: list[np.ndarray],
#             box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
#         image = image[:, :, ::-1]  # RGB -> BGR
#         person_results = process_mmdet_results(det_results, 1)
#         out, _ = inference_top_down_pose_model(self.model,
#                                                image,
#                                                person_results=person_results,
#                                                bbox_thr=box_score_threshold,
#                                                format='xyxy')
#         return out

#     def visualize_pose_results(self,
#                                image: np.ndarray,
#                                pose_results: list[np.ndarray],
#                                kpt_score_threshold: float = 0.3,
#                                vis_dot_radius: int = 4,
#                                vis_line_thickness: int = 1) -> np.ndarray:
#         image = image[:, :, ::-1]  # RGB -> BGR
#         vis = vis_pose_result(self.model,
#                               image,
#                               pose_results,
#                               kpt_score_thr=kpt_score_threshold,
#                               radius=vis_dot_radius,
#                               thickness=vis_line_thickness)
#         return vis[:, :, ::-1]  # BGR -> RGB