import os
import cv2
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import math

# import general_config
from general_config import general_cfg

import sys
# sys.path.insert(0, os.path.abspath('../../opensources/mmpose'))
# print(sys.path)
# print(general_cfg.PROJECT_ROOT)
sys.path.insert(0, os.path.abspath(general_cfg.PROJECT_ROOT + '/model_components/opensources/mmpose'))
# print(sys.path)

# from model_components.opensources.mmpose.mmpose.datasets import Compose
from mmpose.datasets import Compose
from model_components.opensources.mmpose.mmpose.apis import init_pose_model
from mmcv.parallel import collate
from mmcv.image import imwrite

from model_components.configs.pose_est_2d_img import mmpose_topdown_cfg
from model_components.pose_est_2d_img.modeling.build import POSE_EST_2D_IMG_REGISTRY

@POSE_EST_2D_IMG_REGISTRY.register()
def build_mmpose_topdown():
    return mmpose_topdown()

class mmpose_topdown:
    def __init__(self, cfg=mmpose_topdown_cfg.clone()):
        self.cfg = cfg
        self.cfg.freeze()

    def imshow_keypoints(img,
                         pose_result,
                         skeleton=mmpose_topdown_cfg.skeleton,
                         kpt_score_thr=0.3,
                         pose_kpt_color=None,
                         pose_link_color=None,
                         radius=4,
                         thickness=1,
                         show_keypoint_weight=False):
        """Draw keypoints and links on an image.

        Args:
                img (str or Tensor): The image to draw poses on. If an image array
                    is given, id will be modified in-place.
                pose_result (list[kpts]): The poses to draw. Each element kpts is
                    a set of K keypoints as an Kx3 numpy.ndarray, where each
                    keypoint is represented as x, y, score.
                kpt_score_thr (float, optional): Minimum score of keypoints
                    to be shown. Default: 0.3.
                pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                    the keypoint will not be drawn.
                pose_link_color (np.array[Mx3]): Color of M links. If None, the
                    links will not be drawn.
                thickness (int): Thickness of lines.
        """

        # img = mmcv.imread(img)
        # print(img)img
        img_h, img_w, _ = img.shape
        pose_result = [pose_result]
        for kpts in pose_result:
            # draw each point on image
            if pose_kpt_color is not None:
                assert len(pose_kpt_color) == len(kpts)
                for kid, kpt in enumerate(kpts):
                    x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                    if kpt_score > kpt_score_thr:
                        if show_keypoint_weight:
                            img_copy = img.copy()
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                       radius, (int(r), int(g), int(b)), -1)
                            transparency = max(0, min(1, kpt_score))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)
                        else:
                            r, g, b = pose_kpt_color[kid]
                            cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                                       (int(r), int(g), int(b)), -1)

            # draw links
            if skeleton is not None and pose_link_color is not None:
                assert len(pose_link_color) == len(skeleton)
                for sk_id, sk in enumerate(skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                            and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                            and pos2[1] > 0 and pos2[1] < img_h
                            and kpts[sk[0], 2] > kpt_score_thr
                            and kpts[sk[1], 2] > kpt_score_thr):
                        r, g, b = pose_link_color[sk_id]
                        if show_keypoint_weight:
                            img_copy = img.copy()
                            X = (pos1[0], pos2[0])
                            Y = (pos1[1], pos2[1])
                            mX = np.mean(X)
                            mY = np.mean(Y)
                            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                            angle = math.degrees(
                                math.atan2(Y[0] - Y[1], X[0] - X[1]))
                            stickwidth = 2
                            polygon = cv2.ellipse2Poly(
                                (int(mX), int(mY)),
                                (int(length / 2), int(stickwidth)), int(angle), 0,
                                360, 1)
                            cv2.fillConvexPoly(img_copy, polygon,
                                               (int(r), int(g), int(b)))
                            transparency = max(
                                0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                            cv2.addWeighted(
                                img_copy,
                                transparency,
                                img,
                                1 - transparency,
                                0,
                                dst=img)
                        else:
                            cv2.line(
                                img,
                                pos1,
                                pos2, (int(r), int(g), int(b)),
                                thickness=thickness)
        return img

    def show_result(img,
                    pose_result,
                    kpt_score_thr=0.3,
                    pose_kpt_color=None,
                    pose_link_color=None,
                    radius=4,
                    thickness=1,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (list[dict]): The results to draw over `img`
                (bbox_result, pose_result).
            skeleton (list[list]): The connection of keypoints.
                skeleton is 0-based indexing.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints.
                If None, do not draw keypoints.
            pose_link_color (np.array[Mx3]): Color of M links.
                If None, do not draw links.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            radius (int): Radius of circles.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            show (bool): Whether to show the image. Default: False.
            show_keypoint_weight (bool): Whether to change the transparency
                using the predicted confidence scores of keypoints.
            wait_time (int): Value of waitKey param.
                Default: 0.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            Tensor: Visualized img, only if not `show` or `out_file`.
        """
        # img = mmcv.imread(img)
        img = img.copy()

        mmpose_topdown.imshow_keypoints(img, pose_result, mmpose_topdown_cfg.skeleton, kpt_score_thr,
                         pose_kpt_color, pose_link_color, radius,
                         thickness)

        if out_file is not None:
            imwrite(img, out_file)

        return img

    def _box2cs(img_size, box):
        """This encodes bbox(x,y,w,h) into (center, scale)

        Args:
            x, y, w, h

        Returns:
            tuple: A tuple containing center and scale.

            - np.ndarray[float32](2,): Center of the bbox (x, y).
            - np.ndarray[float32](2,): Scale of the bbox w & h.
        """

        x, y, w, h = box[:4]
        input_size = img_size
        aspect_ratio = input_size[0] / input_size[1]
        center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        # pixel std is 200.0
        scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

        scale = scale * 1.25

        return center, scale

    class LoadImage:
        """A simple pipeline to load image."""

        def __init__(self, color_type='color', channel_order='rgb'):
            self.color_type = color_type
            self.channel_order = channel_order

        def __call__(self, results):
            """Call function to load images into results.

            Args:
                results (dict): A result dict contains the img_or_path.

            Returns:
                dict: ``results`` will be returned containing loaded image.
            """
            if isinstance(results['img_or_path'], str):
                results['image_file'] = results['img_or_path']
                img = mmcv.imread(results['img_or_path'], self.color_type,
                                  self.channel_order)
            elif isinstance(results['img_or_path'], np.ndarray):
                results['image_file'] = ''
                if self.color_type == 'color' and self.channel_order == 'rgb':
                    img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)
                else:
                    img = results['img_or_path']
            else:
                raise TypeError('"img_or_path" must be a numpy array or a str or '
                                'a pathlib.Path object')

            results['img'] = img
            return results

    def run(self, imgs, return_heatmap=True):
        channel_order = 'rgb'
        test_pipeline = [mmpose_topdown.LoadImage(channel_order=channel_order)
                         ] + mmpose_topdown_cfg.test_pipline[1:]
        test_pipeline = Compose(test_pipeline)

        # project_root = '/home/xzy/projects/mmpose'
        # project_root = '/home/nk/mmpose/'
        # pose_config = os.path.join(project_root, 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py')
        # pose_checkpoint = os.path.join(project_root, 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')
        # device_ = 'cuda:0'

        # img = os.path.join(mmpose_topdown_cfg.PROJECT_ROOT, img)
        # img_names = os.listdir(img)
        batch_data = []
        # print(self.cfg.img_names)

        for img in imgs:
            # print(img_name)
            # img_path = os.path.join(self.cfg.PROJECT_ROOT, 'test_img', img_name)
            img0_size = img.shape
            bbox = [0, 0, img0_size[1] - 1, img0_size[0] - 1]  # x,y,w,h
            center, scale = mmpose_topdown._box2cs(self.cfg.image_size, bbox)

            data = {
                'img_or_path':
                    img,
                'center':
                    center,
                'scale':
                    scale,
                'bbox_score':
                    1,
                'bbox_id':
                    0,  # need to be assigned if batch_size > 1
                'dataset':
                    None,
                'joints_3d':
                    np.zeros((17, 3), dtype=np.float32),
                'joints_3d_visible':
                    np.zeros((17, 3), dtype=np.float32),
                'rotation':
                    0,
                'ann_info': {
                    'image_size': np.array(self.cfg.image_size),
                    'num_joints': 17,
                    'flip_pairs': None
                }
            }

            data = test_pipeline(data)
            batch_data.append(data)
        batch_data = collate(batch_data, samples_per_gpu=1)

        batch_data['img'] = batch_data['img'].to(self.cfg.DEVICE)
        batch_data['img_metas'] = [
            img_metas[0] for img_metas in batch_data['img_metas'].data
        ]


        pose_model = init_pose_model(
            self.cfg.POSE_CONFIG, self.cfg.POSE_CHECKPOINT, device=self.cfg.DEVICE)

        res = pose_model(batch_data['img'],
                         target=None,
                         target_weight=None,
                         img_metas=batch_data['img_metas'],
                         return_loss=False,
                         return_heatmap=True
                         )

        # print(res.keys())
        # print(res['output_heatmap'].shape)
        # print((res['preds']).shape)

        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])
        pose_link_color = palette[[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ]]
        pose_kpt_color = palette[[
            16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
        ]]
        for i in range(len(res['preds'])):
            # img_name = res['image_paths'][i].split('/')[-1]
            # img_name = img_name.split('.')[0]

            mmpose_topdown.show_result(imgs[i],
                        res['preds'][i],
                        kpt_score_thr=0.3,
                        pose_kpt_color=pose_kpt_color,
                        pose_link_color=pose_link_color,
                        radius=16,
                        thickness=7,
                        out_file='/home/nk/mmpose/outputs/' + str(i) + '.jpg'
                        )

            heatmap = res['output_heatmap'][i]
            heatmap = heatmap * 255
            heatmap = heatmap.astype(int)
            heatmap_total = np.zeros((64, 48))
            # for i in range(heatmap.shape[0]):
            #     heatmap_total = heatmap_total + heatmap[i]
            # plt.imsave('/home/nk/mmpose/outputs/' + img_name + '_heatmap.jpg', heatmap_total, cmap='gray')
            if return_heatmap :
                return res['preds'], heatmap
            else:
                return res['preds']

if __name__ == '__main__':
    img_path ="/home/nk/mmpose/test_img/0_14.jpg"
    # img = mmcv.imread(img_path)
    # print(type(img))
    # print(img.shape)
    img = [cv2.imread(img_path)]
    # print(img)
    # print(img)
    det = mmpose_topdown()
    pred, heatmap = det.run(img)
    print(pred)
