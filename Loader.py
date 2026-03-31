import json


class Loader:
    def __init__(self, config_path="config.json"):
        cfg = json.load(open(config_path, "r"))
        self.model_path = cfg["model_path"]
        self.video_source = cfg["video_path"]
        self.output_segmentation_masks = cfg.get("output_segmentation_masks", False)
        self.min_pose_detection_confidence = cfg.get("min_pose_detection_confidence", 0.5)
        self.min_pose_tracking_confidence = cfg.get("min_pose_tracking_confidence", 0.5)
        self.min_pose_presence_confidence = cfg.get("min_pose_presence_confidence", 0.5)
        self.start_frame = cfg.get("start_frame", 0)
        self.end_frame = cfg.get("end_frame", None)
        self.show_biomechanical_data = cfg.get("show_biomechanical_data", True)
        self.video_output = cfg.get("video_output", True)

    #load the data from the config file and return it as a tuple
    
    def load(self):
        
        return (
            self.model_path,                            #model path
            self.video_source,                          #video path
            self.output_segmentation_masks,             #output segmentation masks(default False)
            self.min_pose_detection_confidence,         #minimum pose detection confidence(default 0.5)
            self.min_pose_tracking_confidence,          #minimum pose tracking confidence(default 0.5)
            self.min_pose_presence_confidence,          #minimum pose presence confidence(default 0.5)
            self.start_frame,                           #start frame(default 0)
            self.end_frame,                             #end frame(default None)
            self.show_biomechanical_data,               #show biomechanical data (default True)
            self.video_output                            #video output (default True)
        )
    