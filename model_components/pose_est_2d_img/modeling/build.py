from fvcore.common.registry import Registry


DETECTION_REGISTRY = Registry('KEYPOINT_DETECTION')

def build_keypoint_detection_model(cfg):
	model_name = cfg.MODEL_NAME
	model = DETECTION_REGISTRY.get(model_name)()
	return model