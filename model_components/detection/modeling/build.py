from fvcore.common.registry import Registry


DETECTION_REGISTRY = Registry('DETECTION')

def build_det_model(cfg):
	model_name = cfg.MODEL_NAME
	model = DETECTION_REGISTRY.get(model_name)()
	return model

