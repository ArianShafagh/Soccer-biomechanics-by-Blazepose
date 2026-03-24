import numpy as np

'''
    here we just calculate the biomechanics of the pose based on the landmarks data and store it in a dictionary called bio_data.
    The biomechanics data can include things like joint angles, limb lengths, and other relevant measurements that
    can be derived from the landmarks data. This is a placeholder for the actual calculations, which would depend on the specific biomechanics you want to analyze.
    For example, you could calculate the angle of the elbow joint by using the coordinates of the shoulder, elbow,
      and wrist landmarks. You could also calculate the length of 
    the upper arm by using the coordinates of the shoulder and elbow landmarks.
    The specific biomechanics you calculate would depend on the use case and the type of analysis you want
    to perform on the pose data.
    Landmarks:
        0 - nose
        1 - left eye (inner)
        2 - left eye
        3 - left eye (outer)
        4 - right eye (inner)
        5 - right eye
        6 - right eye (outer)
        7 - left ear
        8 - right ear
        9 - mouth (left)
        10 - mouth (right)
        11 - left shoulder
        12 - right shoulder
        13 - left elbow
        14 - right elbow
        15 - left wrist
        16 - right wrist
        17 - left pinky
        18 - right pinky
        19 - left index
        20 - right index
        21 - left thumb
        22 - right thumb
        23 - left hip
        24 - right hip
        25 - left knee
        26 - right knee
        27 - left ankle
        28 - right ankle
        29 - left heel
        30 - right heel
        31 - left foot index
        32 - right foot index
        
'''


class BiomechanicalModel:

    def __init__(self, landmarks_data):
        self.landmarks_data = landmarks_data
        self.bio_data = {
            "angles": {},
            "symmetry": {},
            "alignment": {},
        }

    def calculate_angle(self, h, k, a):
        '''
        Calculates the angle between three points.
        h, k, a = [x, y] coordinates for the three points.
        Returns:
            float: The angle in degrees.

        '''
        h,k,a = np.array(h), np.array(k), np.array(a)

        if None in h or None in k or None in a:
            return None  # If any of the points are missing, return None

        # Create Vectors originating from the middle point (k)
        hk = h - k
        ak = a - k

        # Calculate lengths (magnitudes)
        len_hk = np.linalg.norm(hk)
        len_ak = np.linalg.norm(ak)

        if len_hk < 1e-5 or len_ak < 1e-5:
            return None # Avoid division by zero if landmarks are missing
        
        # Calculate the angle in radians, then degrees
        angle_rad = np.arccos(np.dot(hk, ak) / (len_hk * len_ak))
        angle_deg = np.degrees(angle_rad)

        return angle_deg
    

    def knee_valgus_angle(self, h, k, a):
        """
        Calculates horizontal deviation of the knee from the Hip-Ankle line.
        Returns:
        > 0 : Knee is to the RIGHT of the line (Varus/Bow-legged for Left Leg)
        < 0 : Knee is to the LEFT of the line (Valgus/Knock-kneed for Left Leg)
        0   : Perfect Alignment
        """
        if None in h or None in k or None in a:
            return None  
        h_x, h_y = h[0], h[1]
        k_x, k_y = k[0], k[1]
        a_x, a_y = a[0], a[1]

        # Prevent division by zero if hip and ankle are at same height (rare)
        if abs(a_y - h_y) < 1e-5:
            return 0.0

        # 1. Calculate Slope of the leg (Change in X / Change in Y)
        slope = (a_x - h_x) / (a_y - h_y)

        # 2. Calculate where the knee SHOULD be horizontally at its current vertical height
        # Equation of a line: x = x1 + slope * (y - y1)
        expected_knee_x = h_x + slope * (k_y - h_y)

        # 3. Calculate the difference
        deviation = k_x - expected_knee_x

        return deviation

    def analyze(self):

        # ======================= landmarks extraction ========================== #
        # Extracting relevant landmarks for biomechanical analysis.
        
        left_elbow = self.landmarks_data["landmark_13"] if "landmark_13" in self.landmarks_data else {"x": None, "y": None, "z": None}
        right_elbow = self.landmarks_data["landmark_14"] if "landmark_14" in self.landmarks_data else {"x": None, "y": None, "z": None}

        left_shoulder = self.landmarks_data["landmark_11"] if "landmark_11" in self.landmarks_data else {"x": None, "y": None, "z": None}
        right_shoulder = self.landmarks_data["landmark_12"] if "landmark_12" in self.landmarks_data else {"x": None, "y": None, "z": None}

        left_wrist = self.landmarks_data["landmark_15"] if "landmark_15" in self.landmarks_data else {"x": None, "y": None, "z": None}
        right_wrist = self.landmarks_data["landmark_16"] if "landmark_16" in self.landmarks_data else {"x": None, "y": None, "z": None}

        left_hip = self.landmarks_data["landmark_23"] if "landmark_23" in self.landmarks_data else {"x": None, "y": None, "z": None}
        right_hip = self.landmarks_data["landmark_24"] if "landmark_24" in self.landmarks_data else {"x": None, "y": None, "z": None}

        left_knee = self.landmarks_data["landmark_25"] if "landmark_25" in self.landmarks_data else {"x": None, "y": None, "z": None}
        right_knee = self.landmarks_data["landmark_26"] if "landmark_26" in self.landmarks_data else {"x": None, "y": None, "z": None}

        left_ankle = self.landmarks_data["landmark_27"] if "landmark_27" in self.landmarks_data else {"x": None, "y": None, "z": None}
        right_ankle = self.landmarks_data["landmark_28"] if "landmark_28" in self.landmarks_data else {"x": None, "y": None, "z": None}

        left_foot_index = self.landmarks_data["landmark_31"] if "landmark_31" in self.landmarks_data else {"x": None, "y": None, "z": None} 
        right_foot_index = self.landmarks_data["landmark_32"] if "landmark_32" in self.landmarks_data else {"x": None, "y": None, "z": None}



        # ======================= angles ========================== #

        # calculate angles for elbows, knees, shoulders, hips, and ankles
        # What it is: The degree of bend at a specific joint (e.g., how bent is the knee?).

        self.bio_data["angles"]["left_elbow_angle"] = self.calculate_angle(
            [left_shoulder["x"], left_shoulder["y"], left_shoulder["z"]],
            [left_elbow["x"], left_elbow["y"], left_elbow["z"]],
            [left_wrist["x"], left_wrist["y"], left_wrist["z"]]
        )

        self.bio_data["angles"]["right_elbow_angle"] = self.calculate_angle(
            [right_shoulder["x"], right_shoulder["y"], right_shoulder["z"]],
            [right_elbow["x"], right_elbow["y"], right_elbow["z"]],
            [right_wrist["x"], right_wrist["y"], right_wrist["z"]]
        )

        self.bio_data["angles"]["left_knee_angle"] = self.calculate_angle(
            [left_hip["x"], left_hip["y"], left_hip["z"]],
            [left_knee["x"], left_knee["y"], left_knee["z"]],
            [left_ankle["x"], left_ankle["y"], left_ankle["z"]]
        )

        self.bio_data["angles"]["right_knee_angle"] = self.calculate_angle(
            [right_hip["x"], right_hip["y"], right_hip["z"]],
            [right_knee["x"], right_knee["y"], right_knee["z"]],
            [right_ankle["x"], right_ankle["y"], right_ankle["z"]]
        )

        self.bio_data["angles"]["left_shoulder_angle"] = self.calculate_angle(
            [left_elbow["x"], left_elbow["y"], left_elbow["z"]],
            [left_shoulder["x"], left_shoulder["y"], left_shoulder["z"]],
            [left_hip["x"], left_hip["y"], left_hip["z"]]
        )

        self.bio_data["angles"]["right_shoulder_angle"] = self.calculate_angle(
            [right_elbow["x"], right_elbow["y"], right_elbow["z"]],
            [right_shoulder["x"], right_shoulder["y"], right_shoulder["z"]],
            [right_hip["x"], right_hip["y"], right_hip["z"]]
        )

        self.bio_data["angles"]["left_hip_angle"] = self.calculate_angle(
            [left_shoulder["x"], left_shoulder["y"], left_shoulder["z"]],
            [left_hip["x"], left_hip["y"], left_hip["z"]],
            [left_knee["x"], left_knee["y"], left_knee["z"]]
        )

        self.bio_data["angles"]["right_hip_angle"] = self.calculate_angle(
            [right_shoulder["x"], right_shoulder["y"], right_shoulder["z"]],
            [right_hip["x"], right_hip["y"], right_hip["z"]],
            [right_knee["x"], right_knee["y"], right_knee["z"]]
        )

        self.bio_data["angles"]["left_ankle_angle"] = self.calculate_angle(
            [left_knee["x"], left_knee["y"], left_knee["z"]],
            [left_ankle["x"], left_ankle["y"], left_ankle["z"]],
            [left_foot_index["x"], left_foot_index["y"], left_foot_index["z"]]
        )

        self.bio_data["angles"]["right_ankle_angle"] = self.calculate_angle(
            [right_knee["x"], right_knee["y"], right_knee["z"]],
            [right_ankle["x"], right_ankle["y"], right_ankle["z"]],
            [right_foot_index["x"], right_foot_index["y"], right_foot_index["z"]]
        )

        # ======================= Symmetry ========================== #
        # calculating Symmetry for knee,elbow,shoulder,hip and ankle
        # What it is: Comparing the left side of the body to the right side.

        if self.bio_data["angles"]["left_knee_angle"] is not None:
            self.bio_data["symmetry"]["knee_symmetry"] = abs(self.bio_data["angles"]["left_knee_angle"] - self.bio_data["angles"]["right_knee_angle"])  
        else:
            self.bio_data["symmetry"]["knee_symmetry"] = None
        
        if  self.bio_data["angles"]["left_elbow_angle"] is not None:
            self.bio_data["symmetry"]["elbow_symmetry"] = abs(self.bio_data["angles"]["left_elbow_angle"] - self.bio_data["angles"]["right_elbow_angle"])
        else:
            self.bio_data["symmetry"]["elbow_symmetry"] = None
        
        if self.bio_data["angles"]["left_shoulder_angle"] is not None:
            self.bio_data["symmetry"]["shoulder_symmetry"] = abs(self.bio_data["angles"]["left_shoulder_angle"] - self.bio_data["angles"]["right_shoulder_angle"])
        else:
            self.bio_data["symmetry"]["shoulder_symmetry"] = None

        if self.bio_data["angles"]["left_hip_angle"] is not None:
            self.bio_data["symmetry"]["hip_symmetry"] = abs(self.bio_data["angles"]["left_hip_angle"] - self.bio_data["angles"]["right_hip_angle"])
        else:
            self.bio_data["symmetry"]["hip_symmetry"] = None

        if self.bio_data["angles"]["left_ankle_angle"] is not None:
            self.bio_data["symmetry"]["ankle_symmetry"] = abs(self.bio_data["angles"]["left_ankle_angle"] - self.bio_data["angles"]["right_ankle_angle"])
        else:
            self.bio_data["symmetry"]["ankle_symmetry"] = None
        

        # ======================= alignment ========================== #
        # What it is: How body segments line up relative to each other (not just the bend at one joint).
        # It answers: "Is the knee tracking over the toe?" or "Is the trunk leaning too far?"

        # --- A. Trunk Lean (Posture) ---
        if None not in [left_shoulder["x"], left_shoulder["y"], left_shoulder["z"],
                        right_shoulder["x"], right_shoulder["y"], right_shoulder["z"],
                        left_hip["x"], left_hip["y"], left_hip["z"],
                        right_hip["x"], right_hip["y"], right_hip["z"]]:
            
            mid_shldr = [(left_shoulder["x"] + right_shoulder["x"]) / 2, (left_shoulder["y"] + right_shoulder["y"]) / 2, (left_shoulder["z"] + right_shoulder["z"]) / 2]
            
            torso_angle_forward_rad = np.arctan2(mid_shldr[2], mid_shldr[1])  # Z over Y for forward/backward lean
            torso_angle_forward_deg = np.degrees(torso_angle_forward_rad)

            torso_angle_side_rad = np.arctan2(mid_shldr[0], mid_shldr[1])  # X over Y for side lean
            torso_angle_side_deg = np.degrees(torso_angle_side_rad)

            self.bio_data["alignment"]["trunk_lean_forward_deg"] = torso_angle_forward_deg
            self.bio_data["alignment"]["trunk_lean_side_deg"] = torso_angle_side_deg
            
            # --- B. Knee Valgus Proxy (Frontal Plane Alignment) ---
            # Using the angle between the hip, knee, and ankle to estimate knee valgus (inward collapse of the knee).
            # Note: This is a proxy and not a direct measure of valgus, but it can indicate potential alignment issues.

            self.bio_data["alignment"]["left_knee_valgus_proxy"] = self.knee_valgus_angle(
                [left_hip["x"], left_hip["y"]],
                [left_knee["x"], left_knee["y"]],
                [left_ankle["x"], left_ankle["y"]]
            )

            self.bio_data["alignment"]["right_knee_valgus_proxy"] = self.knee_valgus_angle(
                [right_hip["x"], right_hip["y"]],
                [right_knee["x"], right_knee["y"]],
                [right_ankle["x"], right_ankle["y"]]
            )
               
        else:
            self.bio_data["alignment"]["trunk_lean_forward_deg"] = None
            self.bio_data["alignment"]["trunk_lean_side_deg"] = None
            self.bio_data["alignment"]["left_knee_valgus_proxy"] = None
            self.bio_data["alignment"]["right_knee_valgus_proxy"] = None

        return self.bio_data
        
        



