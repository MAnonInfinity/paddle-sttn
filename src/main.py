import torch
import shutil
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys
from functools import cached_property

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.tools.common_tools import is_video_or_image, is_image_file
from src.scenedetect import scene_detect
from src.scenedetect.detectors import ContentDetector
from src.inpaint.sttn_inpaint import STTNInpaint, STTNVideoInpaint
from src.inpaint.lama_inpaint import LamaInpaint
from src.inpaint.video_inpaint import VideoInpaint
from src.tools.inpaint_tools import create_mask, batch_generator
import importlib
import platform
import tempfile
import multiprocessing
from shapely.geometry import Polygon
import time
from tqdm import tqdm

# --- CONFIGURATION ---
VIDEO_PATH = "videos/8.mp4"
OUTPUT_VIDEO = "output.mp4"

# OCR language for subtitle detection.
# 'ch'  = Chinese + English (default, works for CJK videos)
# 'en'  = English-only (use this for videos 1, 2, 8 if they have English subtitles)
# 'latin' = Latin-script languages (French, Spanish, etc.)
OCR_LANG = "ch"

# Subtitle area override — set to (ymin, ymax, xmin, xmax) to force-process a region
# even when OCR detects nothing.  Leave as None for automatic detection.
# Example for 1920x1080: SUB_AREA = (900, 1060, 80, 1840)
SUB_AREA = None
# ---------------------


class SubtitleDetect:
    """
    Text box detection class, used to detect if there are text boxes in video frames
    """

    def __init__(self, video_path, sub_area=None):
        self.video_path = video_path
        self.sub_area = sub_area

    @cached_property
    def text_detector(self):
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import PaddleOCR
        importlib.reload(config)
        # Use the high-level API — it handles model download, GPU, and
        # language selection correctly without needing the local model dir.
        # det_only skips recognition to save time.
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang=OCR_LANG,
            use_gpu=torch.cuda.is_available(),
            show_log=False,
            # Lower thresholds for subtitle detection — defaults (0.3/0.6) are tuned
            # for document OCR and miss many subtitle frames in video
            det_db_thresh=0.2,
            det_db_box_thresh=0.4,
        )
        print(f'[OCR] Using PaddleOCR lang={OCR_LANG!r} with det_db_thresh=0.2, box_thresh=0.4')
        # Return a callable that returns (dt_boxes, elapse) like the old API
        def _detect(img):
            import time
            import numpy as np
            t = time.time()
            result = ocr.ocr(img, cls=False, rec=False)
            boxes = result[0] if result and result[0] else []
            # Convert polygon list → numpy array matching old TextDetector output shape
            dt_boxes = np.array(boxes, dtype=np.float32) if boxes else np.array([], dtype=np.float32).reshape(0, 4, 2)
            return dt_boxes, time.time() - t
        return _detect

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse

    @staticmethod
    def get_coordinates(dt_box):
        """
        Get coordinates from the returned detection box
        :param dt_box: result returned from detection box
        :return: list of coordinate points
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        subtitle_frame_no_box_dict = {}
        print('[Processing] start finding subtitles...')
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # If reading video frame fails (video reaches the last frame)
            if not ret:
                break
            # Successfully read video frame
            current_frame_no += 1
            dt_boxes, elapse = self.detect_subtitle(frame)
            coordinate_list = self.get_coordinates(dt_boxes.tolist())
            if coordinate_list:
                temp_list = []
                for coordinate in coordinate_list:
                    xmin, xmax, ymin, ymax = coordinate
                    if self.sub_area is not None:
                        s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                        if (s_xmin <= xmin and xmax <= s_xmax
                                and s_ymin <= ymin
                                and ymax <= s_ymax):
                            temp_list.append((xmin, xmax, ymin, ymax))
                    else:
                        temp_list.append((xmin, xmax, ymin, ymax))
                if len(temp_list) > 0:
                    subtitle_frame_no_box_dict[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        # if config.UNITE_COORDINATES:
        #     subtitle_frame_no_box_dict = self.get_subtitle_frame_no_box_dict_with_united_coordinates(subtitle_frame_no_box_dict)
        #     if sub_remover is not None:
        #         try:
        #             # When frame count > 1, it indicates it's not a single image or frame
        #             if sub_remover.frame_count > 1:
        #                 subtitle_frame_no_box_dict = self.filter_mistake_sub_area(subtitle_frame_no_box_dict,
        #                                                                           sub_remover.fps)
        #         except Exception:
        #             pass
        #     subtitle_frame_no_box_dict = self.prevent_missed_detection(subtitle_frame_no_box_dict)
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        detected_count = len(new_subtitle_frame_no_box_dict)
        print(f'[Finished] Finished finding subtitles. Detected in {detected_count}/{int(frame_count)} frames.')
        if detected_count == 0:
            print('[Hint] OCR found NO text. Try: OCR_LANG="en", or set STTN_SKIP_DETECTION=True + SUB_AREA=(ymin,ymax,xmin,xmax).')
        else:
            # Print first 3 detected frames so user can see what coordinates were found
            for i, (fn, boxes) in enumerate(list(new_subtitle_frame_no_box_dict.items())[:3]):
                print(f'  [Sample frame {fn}]:')
                for box in boxes:
                    print(f'    -> {box}')

        return new_subtitle_frame_no_box_dict

    def convertToOnnxModelIfNeeded(self, model_dir, model_filename="inference.pdmodel", params_filename="inference.pdiparams", opset_version=14):
        """Converts a Paddle model to ONNX if ONNX providers are available and the model does not already exist."""
        
        if not config.ONNX_PROVIDERS:
            return model_dir
        
        onnx_model_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")
            return onnx_model_path
        
        print(f"Converting Paddle model {model_dir} to ONNX...")
        model_file = os.path.join(model_dir, model_filename)
        params_file = os.path.join(model_dir, params_filename) if params_filename else ""

        try:
            import paddle2onnx
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

            # Convert and save the model
            onnx_model = paddle2onnx.export(
                model_filename=model_file,
                params_filename=params_file,
                save_file=onnx_model_path,
                opset_version=opset_version,
                auto_upgrade_opset=True,
                verbose=True,
                enable_onnx_checker=True,
                enable_experimental_op=True,
                enable_optimize=True,
                custom_op_info={},
                deploy_src="onnxruntime",
                calibration_file="calibration.cache",
                external_file=os.path.join(model_dir, "external_data"),
                export_fp16_model=False,
            )

            print(f"Conversion successful. ONNX model saved to: {onnx_model_path}")
            return onnx_model_path
        except Exception as e:
            print(f"Error during conversion: {e}")
            return model_dir


    @staticmethod
    def split_range_by_scene(intervals, points):
        # Ensure discrete value list is ordered
        points.sort()
        # List to store result intervals
        result_intervals = []
        # Traverse intervals
        for start, end in intervals:
            # Points in the current interval
            current_points = [p for p in points if start <= p <= end]

            # Traverse discrete points in the current interval
            for p in current_points:
                # If the current discrete point is not the start of the interval, add the interval from the start to the number before the discrete point
                if start < p:
                    result_intervals.append((start, p - 1))
                # Update interval start to the current discrete point
                start = p
            # Add the interval from the last discrete point or interval start to the end of the interval
            result_intervals.append((start, end))
        # Output results
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        Get frame numbers where scene switching occurs
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """Determine if two regions are similar."""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.PIXEL_TOLERANCE_X and abs(xmax1 - xmax2) <= config.PIXEL_TOLERANCE_X and \
            abs(ymin1 - ymin2) <= config.PIXEL_TOLERANCE_Y and abs(ymax1 - ymax2) <= config.PIXEL_TOLERANCE_Y

    def unify_regions(self, raw_regions):
        """Unify continuous similar regions, maintaining list structure."""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # Sort keys to ensure they are continuous
            unified_regions = {}

            # Initialize
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # Add a new list to store matched standard intervals
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # If the current interval is similar to the corresponding interval of the previous key, unify them
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # Update unify_value_map to the latest interval value
                unify_value_map[key] = new_unify_values
                last_key = key

            # Pass the final unified results to unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        Get the start and end frame numbers where subtitles appear
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # Initial interval start value

        for i in range(1, len(numbers)):
            # If the difference between current and previous number is more than 1,
            # the previous interval ends, record the start and end of the current interval
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # This number is the endpoint of the current continuous interval
                ranges.append((start, end))
                start = numbers[i]  # Start next continuous interval
        # Add last interval
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # Initial interval start value
        for i in range(1, len(numbers)):
            # If current frame no and previous frame no interval is more than 1,
            # previous interval ends, record start and end of current interval
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # This number is the endpoint of current continuous interval
                ranges.append((start, end))
                start = numbers[i]  # Start next continuous interval
            # If interval is 1 and coordinates are different from previous frame
            # Record start and end of current interval
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # This number is the endpoint of current continuous interval
                    ranges.append((start, end))
                    start = numbers[i]  # Start next continuous interval
        # Add last interval
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def sub_area_to_polygon(sub_area):
        """
        xmin, xmax, ymin, ymax = sub_area
        """
        s_xmin = sub_area[0]
        s_xmax = sub_area[1]
        s_ymin = sub_area[2]
        s_ymax = sub_area[3]
        return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])

    @staticmethod
    def expand_and_merge_intervals(intervals, expand_size=config.STTN_NEIGHBOR_STRIDE*config.STTN_REFERENCE_LENGTH, max_length=config.STTN_MAX_LOAD_NUM):
        # Initialize output interval list
        expanded_intervals = []

        # Expand each original interval
        for interval in intervals:
            start, end = interval

            # Expand to at least 'expand_size' units, but no more than 'max_length' units
            expansion_amount = max(expand_size - (end - start + 1), 0)

            # Equally distribute expansion before and after original interval
            expand_start = max(start - expansion_amount // 2, 1)  # Ensure start point is at least 1
            expand_end = end + expansion_amount // 2

            # If expanded interval exceeds max length, adjust it
            if (expand_end - expand_start + 1) > max_length:
                expand_end = expand_start + max_length - 1

            # For single points, ensure at least 'expand_size' length
            if start == end:
                if expand_end - expand_start + 1 < expand_size:
                    expand_end = expand_start + expand_size - 1

            # Check for overlap with previous interval and merge accordingly
            if expanded_intervals and expand_start <= expanded_intervals[-1][1]:
                previous_start, previous_end = expanded_intervals.pop()
                expand_start = previous_start
                expand_end = max(expand_end, previous_end)

            # Add expanded interval to result list
            expanded_intervals.append((expand_start, expand_end))

        return expanded_intervals

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length=config.STTN_REFERENCE_LENGTH):
        """
        Merge input subtitle start intervals, ensure min size is STTN_REFERENCE_LENGTH
        """
        expanded = []
        # Process single point intervals separately to expand them
        for start, end in intervals:
            if start == end:  # Single point interval
                # Expand to near target length, ensure no overlap
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = float('inf')
                # Find start of next interval
                for ns, ne in intervals:
                    if ns > end:
                        next_start = ns
                        break
                # Determine new expanded start and end
                new_start = max(start - (target_length - 1) // 2, prev_end + 1)
                new_end = min(start + (target_length - 1) // 2, next_start - 1)
                # If new end is before start, not enough space to expand
                if new_end < new_start:
                    new_start, new_end = start, start  # Keep as is
                expanded.append((new_start, new_end))
            else:
                # Keep non-single point intervals, handle overlaps later
                expanded.append((start, end))
        # Sort to merge intervals overlapping due to expansion
        expanded.sort(key=lambda x: x[0])
        # Merge overlapping intervals if they truly overlap and are less than target length
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            # Check for overlap
            if start <= last_end and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # Need to merge
                merged[-1] = (last_start, max(last_end, end))  # Merge interval
            elif start == last_end + 1 and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # Scenarios where adjacent intervals also need merging
                merged[-1] = (last_start, end)
            else:
                # If no overlap and both exceed target length, keep them
                merged.append((start, end))
        return merged

    def compute_iou(self, box1, box2):
        box1_polygon = self.sub_area_to_polygon(box1)
        box2_polygon = self.sub_area_to_polygon(box2)
        intersection = box1_polygon.intersection(box2_polygon)
        if intersection.is_empty:
            return -1
        else:
            union_area = (box1_polygon.area + box2_polygon.area - intersection.area)
            if union_area > 0:
                intersection_area_rate = intersection.area / union_area
            else:
                intersection_area_rate = 0
            return intersection_area_rate

    def get_area_max_box_dict(self, sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        _area_max_box_dict = dict()
        for start_no, end_no in sub_frame_no_list_continuous:
            # Find the text box with the largest area
            current_no = start_no
            # Find the maximum area of the rectangle in the current interval
            area_max_box_list = []
            while current_no <= end_no:
                for coord in subtitle_frame_no_box_dict[current_no]:
                    # Get coordinates of each text box
                    xmin, xmax, ymin, ymax = coord
                    # Calculate the area of current text box
                    current_area = abs(xmax - xmin) * abs(ymax - ymin)
                    # If max box list is empty, current area is max area
                    if len(area_max_box_list) < 1:
                        area_max_box_list.append({
                            'area': current_area,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax
                        })
                    # If list not empty, check if current box is in same region as max box
                    else:
                        has_same_position = False
                        # Traverse each max box, check if current box is in same line and intersects
                        for area_max_box in area_max_box_list:
                            if (area_max_box['ymin'] - config.THRESHOLD_HEIGHT_DIFFERENCE <= ymin
                                    and ymax <= area_max_box['ymax'] + config.THRESHOLD_HEIGHT_DIFFERENCE):
                                if self.compute_iou((xmin, xmax, ymin, ymax), (
                                        area_max_box['xmin'], area_max_box['xmax'], area_max_box['ymin'],
                                        area_max_box['ymax'])) != -1:
                                    # If height difference different
                                    if abs(abs(area_max_box['ymax'] - area_max_box['ymin']) - abs(
                                            ymax - ymin)) < config.THRESHOLD_HEIGHT_DIFFERENCE:
                                        has_same_position = True
                                    # If same line, check if current area is largest
                                    # Determine area size, if current area larger, update max area coordinates for current line
                                    if has_same_position and current_area > area_max_box['area']:
                                        area_max_box['area'] = current_area
                                        area_max_box['xmin'] = xmin
                                        area_max_box['xmax'] = xmax
                                        area_max_box['ymin'] = ymin
                                        area_max_box['ymax'] = ymax
                        # If traversed all max boxes and found new line, add it
                        if not has_same_position:
                            new_large_area = {
                                'area': current_area,
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax
                            }
                            if new_large_area not in area_max_box_list:
                                area_max_box_list.append(new_large_area)
                                break
                current_no += 1
            _area_max_box_list = list()
            for area_max_box in area_max_box_list:
                if area_max_box not in _area_max_box_list:
                    _area_max_box_list.append(area_max_box)
            _area_max_box_dict[f'{start_no}->{end_no}'] = _area_max_box_list
        return _area_max_box_dict

    def get_subtitle_frame_no_box_dict_with_united_coordinates(self, subtitle_frame_no_box_dict):
        """
        Unify text area coordinates across multiple video frames
        """
        subtitle_frame_no_box_dict_with_united_coordinates = dict()
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        area_max_box_dict = self.get_area_max_box_dict(frame_no_list, subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                area_max_box_list = area_max_box_dict[f'{start_no}->{end_no}']
                current_boxes = subtitle_frame_no_box_dict[current_no]
                new_subtitle_frame_no_box_list = []
                for current_box in current_boxes:
                    current_xmin, current_xmax, current_ymin, current_ymax = current_box
                    for max_box in area_max_box_list:
                        large_xmin = max_box['xmin']
                        large_xmax = max_box['xmax']
                        large_ymin = max_box['ymin']
                        large_ymax = max_box['ymax']
                        box1 = (current_xmin, current_xmax, current_ymin, current_ymax)
                        box2 = (large_xmin, large_xmax, large_ymin, large_ymax)
                        res = self.compute_iou(box1, box2)
                        if res != -1:
                            new_subtitle_frame_no_box = (large_xmin, large_xmax, large_ymin, large_ymax)
                            if new_subtitle_frame_no_box not in new_subtitle_frame_no_box_list:
                                new_subtitle_frame_no_box_list.append(new_subtitle_frame_no_box)
                subtitle_frame_no_box_dict_with_united_coordinates[current_no] = new_subtitle_frame_no_box_list
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict_with_united_coordinates

    def prevent_missed_detection(self, subtitle_frame_no_box_dict):
        """
        Add extra text boxes to prevent missed detection
        """
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                if current_no + 1 != end_no and (current_no + 1) in subtitle_frame_no_box_dict.keys():
                    next_box_list = subtitle_frame_no_box_dict[current_no + 1]
                    if set(current_box_list).issubset(set(next_box_list)):
                        subtitle_frame_no_box_dict[current_no] = subtitle_frame_no_box_dict[current_no + 1]
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict

    @staticmethod
    def get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        sub_area_with_frequency = {}
        for start_no, end_no in sub_frame_no_list_continuous:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                for current_box in current_box_list:
                    if str(current_box) not in sub_area_with_frequency.keys():
                        sub_area_with_frequency[f'{current_box}'] = 1
                    else:
                        sub_area_with_frequency[f'{current_box}'] += 1
                current_no += 1
                if current_no > end_no:
                    break
        return sub_area_with_frequency

    def filter_mistake_sub_area(self, subtitle_frame_no_box_dict, fps):
        """
        Filter incorrect subtitle regions
        """
        sub_frame_no_list_continuous = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        sub_area_with_frequency = self.get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict)
        correct_sub_area = []
        for sub_area in sub_area_with_frequency.keys():
            if sub_area_with_frequency[sub_area] >= (fps // 2):
                correct_sub_area.append(sub_area)
            else:
                print(f'drop {sub_area}')
        correct_subtitle_frame_no_box_dict = dict()
        for frame_no in subtitle_frame_no_box_dict.keys():
            current_box_list = subtitle_frame_no_box_dict[frame_no]
            new_box_list = []
            for current_box in current_box_list:
                if str(current_box) in correct_sub_area and current_box not in new_box_list:
                    new_box_list.append(current_box)
            correct_subtitle_frame_no_box_dict[frame_no] = new_box_list
        return correct_subtitle_frame_no_box_dict


class SubtitleRemover:
    def __init__(self, vd_path, sub_area=None, video_out_name=None, gui_mode=False):
        importlib.reload(config)
        # Thread lock
        self.lock = threading.RLock()
        # User-specified subtitle area location
        self.sub_area = sub_area
        # Whether to run in GUI mode
        self.gui_mode = gui_mode
        # Determine if it is an image
        self.is_picture = False
        if is_image_file(str(vd_path)):
            self.sub_area = None
            self.is_picture = True
        # Video path
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # Get video name from video path
        self.vd_name = Path(self.video_path).stem
        # Total number of video frames
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # Video frame rate
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # Video dimensions
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Create subtitle detection object
        self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # Create video temporary object
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
        # Use XVID codec in AVI container — avoids the OpenCV mp4v malformed-header bug
        # on Linux that causes ffmpeg to SIGSEGV when reading the temp file.
        # The final output is still written as .mp4 by the ffmpeg merge step.
        self.video_writer = cv2.VideoWriter(self.video_temp_file.name, cv2.VideoWriter_fourcc(*'XVID'), self.fps, self.size)
        if video_out_name:
            self.video_out_name = video_out_name
        else:
            self.video_out_name = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4')
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_name = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')
        if torch.cuda.is_available():
            print('use GPU for acceleration')
        if config.USE_DML:
            print('use DirectML for acceleration')
            if config.MODE != config.InpaintMode.STTN:
                print('Warning: DirectML acceleration is only available for STTN model. Falling back to CPU for other models.')
        for provider in config.ONNX_PROVIDERS:
            print(f"Detected execution provider: {provider}")


        # Total processing progress
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # Preview frame
        self.preview_frame = None
        # Whether to embed original audio into subtitle-removed video
        self.is_successful_merged = False

    @staticmethod
    def get_coordinates(dt_box):
        """
        Get coordinates from returned detection box
        :param dt_box: Detection box return result
        :return list: List of coordinate points
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        Determine if given frame no is the start
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        Determine if given frame no is the start
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage) // 2
        self.progress_total = 50 + self.progress_remover

    def propainter_mode(self, tbar):
        print('use propainter mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        scene_div_points = self.sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = self.sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                          scene_div_points)
        self.video_inpaint = VideoInpaint(config.PROPAINTER_MAX_LOAD_NUM)
        print('[Processing] start removing subtitles...')
        index = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            index += 1
            # If current frame has no watermark/text, write directly
            if index not in sub_list.keys():
                self.video_writer.write(frame)
                print(f'write frame: {index}')
                self.update_progress(tbar, increment=1)
                continue
            # If has watermark, check if frame is start frame
            else:
                # If start frame, batch inference to end frame
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # print(f'No 1 Current index: {index}')
                    start_frame_no = index
                    print(f'find start: {start_frame_no}')
                    # Find end frame
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # Determine if current frame no is subtitle start position
                    # If end_frame_no is not -1
                    if end_frame_no != -1:
                        print(f'find end: {end_frame_no}')
                        # ************ Read all frames in this interval start ************
                        temp_frames = list()
                        # Add start frame to processing list
                        temp_frames.append(frame)
                        inner_index = 0
                        # Read until end frame
                        while index < end_frame_no:
                            ret, frame = self.video_cap.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ End reading all frames in this interval ************
                        if len(temp_frames) < 1:
                            # Nothing to process, skip
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask = create_mask(self.mask_size, sub_list[index])
                            if self.lama_inpaint is None:
                                self.lama_inpaint = LamaInpaint()
                            inpainted_frame = self.lama_inpaint(frame, single_mask)
                            self.video_writer.write(inpainted_frame)
                            print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # Process read video frames in batches
                            # 1. Get mask for current batch
                            mask = create_mask(self.mask_size, sub_list[start_frame_no])
                            for batch in batch_generator(temp_frames, config.PROPAINTER_MAX_LOAD_NUM):
                                # 2. Call batch inference
                                if len(batch) == 1:
                                    single_mask = create_mask(self.mask_size, sub_list[start_frame_no])
                                    if self.lama_inpaint is None:
                                        self.lama_inpaint = LamaInpaint()
                                    inpainted_frame = self.lama_inpaint(frame, single_mask)
                                    self.video_writer.write(inpainted_frame)
                                    print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                    self.update_progress(tbar, increment=1)
                                elif len(batch) > 1:
                                    inpainted_frames = self.video_inpaint.inpaint(batch, mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        print(f'write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        if self.gui_mode:
                                            self.preview_frame = cv2.hconcat([batch[i], inpainted_frame])
                                self.update_progress(tbar, increment=len(batch))

    def sttn_mode_with_no_detection(self, tbar):
        """
        Use sttn to repaint selected area, no subtitle detection
        """
        print('use sttn mode with no detection')
        print('[Processing] start removing subtitles...')
        if self.sub_area is not None:
            ymin, ymax, xmin, xmax = self.sub_area
        else:
            print('[Info] No subtitle area has been set. Video will be processed in full screen. As a result, the final outcome might be suboptimal.')
            ymin, ymax, xmin, xmax = 0, self.frame_height, 0, self.frame_width
        mask_area_coordinates = [(xmin, xmax, ymin, ymax)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def sttn_mode(self, tbar):
        # Skip searching for subtitle frames?
        if config.STTN_SKIP_DETECTION:
            # If skipped, use sttn mode directly
            self.sttn_mode_with_no_detection(tbar)
        else:
            print('use sttn mode')
            sttn_inpaint = STTNInpaint()
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            if not sub_list:
                if self.sub_area is not None:
                    print('[Info] No subtitles detected by OCR. Processing configured sub_area anyway.')
                    self.sttn_mode_with_no_detection(tbar)
                else:
                    print('[Warning] No subtitles detected and no sub_area is configured.')
                    print('[Warning] Set sub_area in the script or configure STTN_SKIP_DETECTION=True to force-process a region.')
                    print('[Warning] Output video will be a copy of the input (no processing done).')
                return
            continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
            print(f'[Detection] Found {len(sub_list)} subtitle frames in {len(continuous_frame_no_list)} intervals.')
            # Expand intervals to cover neighboring frames that OCR may have missed
            # (subtitles persist across time, so detected frames anchor the regions)
            continuous_frame_no_list = self.sub_detector.expand_and_merge_intervals(continuous_frame_no_list)
            continuous_frame_no_list = self.sub_detector.filter_and_merge_intervals(continuous_frame_no_list)
            print(f'[Detection] After expansion: {len(continuous_frame_no_list)} intervals covering {sum(e - s + 1 for s, e in continuous_frame_no_list)} frames.')
            start_end_map = dict()
            for interval in continuous_frame_no_list:
                start, end = interval
                start_end_map[start] = end
            current_frame_index = 0
            print('[Processing] start removing subtitles...')
            while True:
                ret, frame = self.video_cap.read()
                # If EOF, end
                if not ret:
                    break
                current_frame_index += 1
                # If current frame no is start of subtitle interval. If not, write directly
                if current_frame_index not in start_end_map.keys():
                    self.video_writer.write(frame)
                    print(f'write frame: {current_frame_index}')
                    self.update_progress(tbar, increment=1)
                    if self.gui_mode:
                        self.preview_frame = cv2.hconcat([frame, frame])
                # If start of interval, find end
                else:
                    start_frame_index = current_frame_index
                    end_frame_index = start_end_map[current_frame_index]
                    print(f'processing frame {start_frame_index} to {end_frame_index}')
                    # List to store video frames needing inpaint
                    frames_need_inpaint = list()
                    frames_need_inpaint.append(frame)
                    inner_index = 0
                    # Continue reading until end
                    for j in range(end_frame_index - start_frame_index):
                        ret, frame = self.video_cap.read()
                        if not ret:
                            break
                        current_frame_index += 1
                        frames_need_inpaint.append(frame)
                    mask_area_coordinates = []
                    # 1. Get the set of all mask coordinates for current batch
                    for mask_index in range(start_frame_index, end_frame_index):
                        if mask_index in sub_list.keys():
                            for area in sub_list[mask_index]:
                                xmin, xmax, ymin, ymax = area
                                # Determine if it is a non-subtitle region (if width > length, it is considered error detection)
                                if (ymax - ymin) - (xmax - xmin) > config.THRESHOLD_HEIGHT_WIDTH_DIFFERENCE:
                                    continue
                                if area not in mask_area_coordinates:
                                    mask_area_coordinates.append(area)
                    # 1. Get mask for current batch
                    mask = create_mask(self.mask_size, mask_area_coordinates)
                    print(f'inpaint with mask: {mask_area_coordinates}')
                    for batch in batch_generator(frames_need_inpaint, config.STTN_MAX_LOAD_NUM):
                        # 2. Call batch inference
                        if len(batch) >= 1:
                            inpainted_frames = sttn_inpaint(batch, mask)
                            for i, inpainted_frame in enumerate(inpainted_frames):
                                self.video_writer.write(inpainted_frame)
                                print(f'write frame: {start_frame_index + inner_index} with mask')
                                inner_index += 1
                                if self.gui_mode:
                                    self.preview_frame = cv2.hconcat([batch[i], inpainted_frame])
                        self.update_progress(tbar, increment=len(batch))

    def lama_mode(self, tbar):
        print('use lama mode')
        sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
        if self.lama_inpaint is None:
            self.lama_inpaint = LamaInpaint()
        index = 0
        print('[Processing] start removing subtitles...')
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            original_frame = frame
            index += 1
            if index in sub_list.keys():
                mask = create_mask(self.mask_size, sub_list[index])
                if config.LAMA_SUPER_FAST:
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    frame = self.lama_inpaint(frame, mask)
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, frame])
            if self.is_picture:
                cv2.imencode(self.ext, frame)[1].tofile(self.video_out_name)
            else:
                self.video_writer.write(frame)
            tbar.update(1)
            self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
            self.progress_total = 50 + self.progress_remover

    def run(self):
        start_time = time.time()
        start_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
        print(f'[Started]  {start_dt}')
        # Reset progress bar
        self.progress_total = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')
        if self.is_picture:
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            self.lama_inpaint = LamaInpaint()
            original_frame = cv2.imread(self.video_path)
            if len(sub_list):
                mask = create_mask(original_frame.shape[0:2], sub_list[1])
                inpainted_frame = self.lama_inpaint(original_frame, mask)
            else:
                inpainted_frame = original_frame
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, inpainted_frame])
            cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_name)
            tbar.update(1)
            self.progress_total = 100
        else:
            # In precision mode, get scene segment frame numbers for further cutting
            if config.MODE == config.InpaintMode.PROPAINTER:
                self.propainter_mode(tbar)
            elif config.MODE == config.InpaintMode.STTN:
                self.sttn_mode(tbar)
            else:
                self.lama_mode(tbar)
        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # Merge original audio into newly generated video file
            self.merge_audio_to_video()
            print(f"[Finished]Subtitle successfully removed, video generated at：{self.video_out_name}")
        else:
            print(f"[Finished]Subtitle successfully removed, picture generated at：{self.video_out_name}")
        end_time = time.time()
        end_dt = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
        elapsed = int(end_time - start_time)
        elapsed_str = f'{elapsed // 3600:02d}h {(elapsed % 3600) // 60:02d}m {elapsed % 60:02d}s'
        print(f'[Finished] {end_dt}')
        print(f'[Elapsed]  {elapsed_str}  ({elapsed}s total)')
        self.isFinished = True
        self.progress_total = 100
        # Handle Colab-specific features like auto-download
        self.handle_colab_features()
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(f'failed to delete temp file {self.video_temp_file.name}')

    def handle_colab_features(self):
        """
        Handle Google Colab specific features like auto-download.
        """
        try:
            import google.colab
            is_colab = True
        except ImportError:
            is_colab = False

        if not is_colab:
            return

        # Handle Auto-download
        if config.AUTO_DOWNLOAD:
            from google.colab import files
            import time
            import shutil

            # Resolve to an absolute path so os.path.dirname never returns ""
            # (which happens when video_out_name is a bare filename like "output.mp4")
            abs_out = os.path.abspath(self.video_out_name)
            out_dir = os.path.dirname(abs_out)

            # timestamp format: minute hour date month year -> %M %H %d %m %Y
            timestamp_name = time.strftime('%M %H %d %m %Y.mp4')
            download_path = os.path.join(out_dir, timestamp_name)

            print(f"[Colab] Output video path: {abs_out}")
            print(f"[Colab] Preparing download as: {timestamp_name}")

            if os.path.exists(abs_out):
                shutil.copy2(abs_out, download_path)
                print(f"[Colab] Copied to: {download_path}")
                # Small pause so Colab's output stream can flush before the
                # JS download bridge is invoked — without this the browser
                # download notification sometimes never appears.
                time.sleep(1)
                print(f"[Colab] Triggering browser download...")
                files.download(download_path)
            else:
                print(f"[Error] Output video not found at {abs_out}")

    def merge_audio_to_video(self):
        if not os.path.exists(self.video_temp_file.name):
            print(f'[Error] Processed video not found at temp path: {self.video_temp_file.name}')
            return

        # Single-pass merge: take video stream from processed file, audio from original.
        # -map 1:a? = audio from original is optional (handles videos with no audio track).
        # -c:a aac  = re-encode audio to AAC (works for any input codec: mp3, ac3, pcm...).
        # -c:v copy = keep processed video as-is (no re-encoding overhead) UNLESS h264 needed.
        use_shell = True if os.name == 'nt' else False
        cmd = [
            config.FFMPEG_PATH, '-y',
            '-i', self.video_temp_file.name,  # processed video (no audio)
            '-i', self.video_path,             # original video (audio source)
            '-map', '0:v:0',                   # video from processed
            '-map', '1:a?',                    # audio from original (optional)
            '-c:v', 'libx264' if config.USE_H264 else 'copy',
            '-c:a', 'aac',
            '-loglevel', 'warning',
            self.video_out_name,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    stdin=open(os.devnull), shell=use_shell)
            if result.returncode != 0:
                print(f'[Warning] ffmpeg audio merge failed (exit {result.returncode}):')
                print(result.stderr[:800])
                shutil.copy2(self.video_temp_file.name, self.video_out_name)
            else:
                if result.stderr.strip():
                    print(f'[ffmpeg] {result.stderr.strip()[:400]}')
                self.is_successful_merged = True
                print('[Info] Audio merged successfully.')
        except Exception as e:
            print(f'[Error] Could not run ffmpeg: {e}')
            try:
                shutil.copy2(self.video_temp_file.name, self.video_out_name)
            except IOError as ie:
                print(f'[Error] Unable to copy processed video: {ie}')
        finally:
            self.video_temp_file.close()


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

    if is_video_or_image(VIDEO_PATH):
        sd = SubtitleRemover(VIDEO_PATH, sub_area=SUB_AREA, video_out_name=OUTPUT_VIDEO)
        sd.run()
    else:
        print(f'Invalid video path: {VIDEO_PATH}')
