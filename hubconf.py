dependencies = ['torch']
from modules.xfeat import XFeat as _XFeat
import torch

def XFeat(pretrained=True, top_k=4096, detection_threshold=0.05, custom=False, weights_path=None):
    """
    XFeat model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = None
    if pretrained:
        if not custom:
            weights = torch.hub.load_state_dict_from_url(
                "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt",
                map_location=device
                )
        else:
            weights = torch.load(
                weights_path,
                map_location=device
                )
    
    model = _XFeat(weights, top_k=top_k, detection_threshold=detection_threshold)
    return model
