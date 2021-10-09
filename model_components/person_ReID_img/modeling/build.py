from fvcore.common.registry import Registry


PERSON_REID_IMG_REGISTRY = Registry('PERSON_REID_IMG')

def build_person_reid_img_model(cfg):
	model_name = cfg.MODEL_NAME
	model = PERSON_REID_IMG_REGISTRY.get(model_name)()
	return model

