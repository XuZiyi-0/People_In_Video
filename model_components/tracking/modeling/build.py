from fvcore.common.registry import Registry


TRACKING_REGISTRY = Registry('TRACKING')

def build_trk_model(cfg):
	model_name = cfg.MODEL_NAME
	model = TRACKING_REGISTRY.get(model_name)()
	return model

