from fvcore.common.registry import Registry


IMG_CLS_REGISTRY = Registry('IMG_CLS')

def build_img_cls_model(cfg):
	model_name = cfg.MODEL_NAME
	model = IMG_CLS_REGISTRY.get(model_name)()
	return model