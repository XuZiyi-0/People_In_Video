from fvcore.common.registry import Registry


E2E_TRACKING_REGISTRY = Registry('E2E_TRACKING')

def build_e2e_tracking_model(cfg):
	model_name = cfg.MODEL.NAME
	model = E2E_TRACKING_REGISTRY.get(model_name)(cfg)
	return model