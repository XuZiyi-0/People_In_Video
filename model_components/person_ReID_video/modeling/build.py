from fvcore.common.registry import Registry


PERSON_REID_VIDEO_REGISTRY = Registry('PERSON_REID_VIDEO')

def build_person_reid_video_model(cfg):
	model_name = cfg.MODEL_NAME
	model = PERSON_REID_VIDEO_REGISTRY.get(model_name)()
	return model
