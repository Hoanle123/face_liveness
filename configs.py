"""Load configuration from .ini file."""
import os
from dotenv import load_dotenv
load_dotenv()
    
class ENV:
    
    FACEMODEL_PATH = os.getenv('FACEMODEL_PATH')
    LIVENESS_PATH = os.getenv('LIVENESS_PATH')
    RATIO_THRESHOLD_FACE_SIZE = float(os.getenv('RATIO_THRESHOLD_FACE_SIZE'))
    RATIO_THRESHOLD_STRAIGHT = float(os.getenv('RATIO_THRESHOLD_STRAIGHT'))
    RATIO_TRACKING_BBOX = float(os.getenv('RATIO_TRACKING_BBOX'))

    





