from fvcore.common.registry import Registry
DET_AND_TRC_REGISTRY = Registry('DET_AND_TRC')

def build_det_and_trc_model(cfg):
	model_name = cfg.MODEL.NAME
	model = DET_AND_TRC_REGISTRY.get(model_name)(cfg)
	return model