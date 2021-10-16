from fvcore.common.registry import Registry


POSE_EST_2D_IMG_REGISTRY = Registry('POSE_EST_2D_IMG_REGISTRY')

def build_keypoint_detection_model(cfg):
	model_name = cfg.MODEL_NAME
	model = POSE_EST_2D_IMG_REGISTRY.get(model_name)()
	return model