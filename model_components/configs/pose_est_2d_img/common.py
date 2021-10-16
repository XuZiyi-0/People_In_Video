from fvcore.common.config import CfgNode

keypoint_detection_common_cfg = CfgNode()

keypoint_detection_common_cfg.MODEL_NAME = 'build_mmpose_topdown'
keypoint_detection_common_cfg.DEVICE_ID = 0,