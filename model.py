import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
from biomechanical import BiomechanicalModel
import json


class Blazepose:

    def __init__(self, model_path, video_path, 
                 output_segmentation_masks=False, 
                 min_pose_detection_confidence=0.5, 
                 min_tracking_confidence=0.5, 
                 min_pose_presence_confidence=0.5, 
                 start_frame=0,
                 end_frame=None,
                 show_biomechanical_data=True):
        
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            running_mode = running_mode_lib.VisionTaskRunningMode.VIDEO,
            base_options=base_options,
            output_segmentation_masks=output_segmentation_masks,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.video_path = video_path
        self.landmarks_data_pixel = {}
        self.landmarks_data_world = {}
        self.bio_data = {}
        self.frame_idx = 0
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.show_biomechanical_data = show_biomechanical_data
        self.paused = False

    def get_landmark_coords_pixel(self, detection_result, landmark_index, image_width, image_height):
        """
        Get pixel coordinates for a specific landmark
        """
        if not detection_result.pose_landmarks:
            return None
        
        if landmark_index >= len(detection_result.pose_landmarks[0]):
            return None
        
        # Get first detected person
        pose_landmarks = detection_result.pose_landmarks[0]
        
        # Get specific landmark
        landmark = pose_landmarks[landmark_index]

        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        z = landmark.z
        vis = landmark.visibility
        result={"x": x, "y": y, "z": z, "visibility": vis}

        return result

    def get_landmark_coords_world(self, detection_result, landmark_index):
        """
        Get world coordinates for a specific landmark
        """
        if not detection_result.pose_world_landmarks:
            return None
        
        if landmark_index >= len(detection_result.pose_world_landmarks[0]):
            return None
        
        # Get first detected person
        pose_landmarks = detection_result.pose_world_landmarks[0]
        
        # Get specific landmark
        landmark = pose_landmarks[landmark_index]

        x = landmark.x
        y = landmark.y
        z = landmark.z
        vis = landmark.visibility
        result={"x": x, "y": y, "z": z, "visibility": vis}

        return result

    def draw_landmarks_on_image(self, frame, pose_landmarks_pixel_list):
        annotated_image = np.copy(frame)
        if not pose_landmarks_pixel_list:
            return annotated_image
        
        pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
        pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_landmarks in pose_landmarks_pixel_list:
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=pose_landmarks,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_landmark_style,
                connection_drawing_spec=pose_connection_style)

        return annotated_image
    
    def detect(self, frame, timestamp):
        image_height, image_width, _ = frame.shape
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = self.detector.detect_for_video(image_mp, timestamp)

        return detection_result,image_height, image_width
    
    def save_all_data(self):
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open("output\\landmarks_data_world.json", "w") as f:
            json.dump(self.landmarks_data_world, f, indent=4)
        with open("output\\biomechanical_analysis_data.json", "w") as f:
            json.dump(self.bio_data, f, indent=4)
        with open("output\\landmarks_data_pixel.json", "w") as f:
            json.dump(self.landmarks_data_pixel, f, indent=4)

    def save_annotated_video(self, cap):
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter("output\\VideoProject_annotated.mp4", fourcc, fps, (width, height))

    def extract_landmarks_pixel(self, detection_result, image_width, image_height):
        landmarks_data = {}
        for idx in range(33):  # Blazepose has 33 landmarks
            coords = self.get_landmark_coords_pixel(detection_result, idx, image_width, image_height)
            if coords:
                landmarks_data[f'landmark_{idx}'] = coords
        return landmarks_data
    
    def extract_landmarks_world(self, detection_result):
        landmarks_data = {}
        for idx in range(33):  # Blazepose has 33 landmarks
            coords = self.get_landmark_coords_world(detection_result, idx)
            if coords:
                landmarks_data[f'landmark_{idx}'] = coords
        return landmarks_data
    
    def tracking_point(self, pose_landmarks_pixel_list):
        if not pose_landmarks_pixel_list:
            return {'x': None, 'y': None}

        pixel_left_x,pixel_left_y = pose_landmarks_pixel_list['landmark_23']["x"],pose_landmarks_pixel_list['landmark_23']["y"]
        pixel_right_x,pixel_right_y = pose_landmarks_pixel_list['landmark_24']["x"],pose_landmarks_pixel_list['landmark_24']["y"]

        pixel_tracking_x = int((pixel_left_x + pixel_right_x) / 2)
        pixel_tracking_y = int((pixel_left_y + pixel_right_y) / 2)
        return {'x': pixel_tracking_x, 'y': pixel_tracking_y}

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        output_video = self.save_annotated_video(cap)
        print("Starting video processing...")
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.frame_idx = self.start_frame
        start_time = time.time()
        while cap.isOpened():
            if not self.paused:
                success, frame = cap.read()
                if not success or (self.end_frame is not None and self.frame_idx >= self.end_frame):
                    print("End of video.")
                    break
                
                timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                
                # Perform detection and analysis
                detection_result,image_height, image_width = self.detect(frame, timestamp)


                # Extract and save landmarks
                landmarks_pixel = self.extract_landmarks_pixel(detection_result, image_width, image_height)
                landmarks_world = self.extract_landmarks_world(detection_result)
                self.landmarks_data_pixel[self.frame_idx] = landmarks_pixel
                self.landmarks_data_world[self.frame_idx] = landmarks_world
        
                # drawing the landmarks on the image and saving the annotated image to a variable
                annotated_image = self.draw_landmarks_on_image(frame, detection_result.pose_landmarks)

                # save the annotated video
                output_video.write(annotated_image)

                # Extract and save biomechanical analysis results
                bio_result = BiomechanicalModel(landmarks_world).analyze()
                self.bio_data[self.frame_idx] = bio_result

                tracking_point = self.tracking_point(landmarks_pixel)
                cv2.circle(annotated_image, (tracking_point["x"], tracking_point["y"]), 10, (255, 0, 0), -1)

                if self.show_biomechanical_data:
                    trunck_lean_angle = bio_result["alignment"]["trunk_lean_forward_deg"]
                    left_knee_valgus_proxy = bio_result["alignment"]["left_knee_valgus_proxy"]
                    right_knee_valgus_proxy = bio_result["alignment"]["right_knee_valgus_proxy"]
                    trunk_side_lean_angle = bio_result["alignment"]["trunk_lean_side_deg"]
                    cv2.putText(annotated_image, f'Trunck forward lean angle: {trunck_lean_angle}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_image, f'Trunck side lean angle: {trunk_side_lean_angle}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_image, f'left knee valgus proxy: {left_knee_valgus_proxy}', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_image, f'right knee valgus proxy: {right_knee_valgus_proxy}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_image, f'Frame: {self.frame_idx}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            # control the video playback with 'q' to quit and space to pause/resume
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting video processing.")
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")
            
            if self.paused:
                cv2.putText(annotated_image, 'PAUSED', (image_width//2, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                self.frame_idx += 1


            # Display the resulting frame
            cv2.imshow('Blazepose Detection', annotated_image)

        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds and avg fps is :{self.frame_idx/total_time:.2f}")        
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

        self.save_all_data()
        print("Data saved successfully.")
        print("Processing complete and terminated successfully.")




