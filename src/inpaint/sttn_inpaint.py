import copy
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config
from src.inpaint.sttn.auto_sttn import InpaintGenerator
from src.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor

# Define image preprocessing
_to_tensors = transforms.Compose([
    Stack(),  # Stack images into sequence
    ToTorchFormatTensor()  # Convert stacked images to PyTorch tensor
])


class STTNInpaint:
    def __init__(self):
        self.device = config.device
        # 1. Create InpaintGenerator model instance and load onto chosen device
        self.model = InpaintGenerator().to(self.device)
        # 2. Load pre-trained model weights, load model state dict
        self.model.load_state_dict(torch.load(config.STTN_MODEL_PATH, map_location='cpu')['netG'])
        # 3. Set model to evaluation mode
        self.model.eval()
        # Width and height used for model input
        self.model_input_width, self.model_input_height = 640, 120
        # 2. Set neighbor stride
        self.neighbor_stride = config.STTN_NEIGHBOR_STRIDE
        self.ref_length = config.STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: original video frames
        :param input_mask: subtitle region mask
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # Determine vertical height for subtitle removal
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)
        # Initialize frame storage variables
        # High resolution frame storage list
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}  # Dictionary to store scaled frames
        comps = {}  # Dictionary to store completed frames
        # Storage for final video frames
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # Initialize list for each removal part

        # Read and scale frames
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # Crop and scale for each removal part
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # Crop
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # Resize
                frames_scaled[k].append(image_resize)  # Add resized frame to corresponding list

        # Process each removal part
        for k in range(len(inpaint_area)):
            # Call inpaint function for processing
            comps[k] = self.inpaint(frames_scaled[k])

        # If removal parts exist
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # Get original frame
                # For each segment in the pattern
                for k in range(len(inpaint_area)):
                    custom_h = inpaint_area[k][1] - inpaint_area[k][0]
                    comp = cv2.resize(comps[k][j], (W_ori, custom_h))  # Resize completed frame back to original size
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)  # Convert color space
                    # Get mask area and perform image composition
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]  # Get mask area
                    # Implement image fusion within the mask area
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                # Add final frame to list
                inpainted_frames.append(frame)
        else:
            # If no inpaint areas found, pass frames through unchanged
            inpainted_frames = copy.deepcopy(input_frames)
            
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # Convert to binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        Sample reference frames for the entire video
        """
        # Initialize reference frame index list
        ref_index = []
        # Iterate through video length based on ref_length
        for i in range(0, length, self.ref_length):
            # If current frame is not in neighbor frames
            if i not in neighbor_ids:
                # Add it to reference frames list
                ref_index.append(i)
        # Return reference frame index list
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        Use STTN to complete hole filling (holes are masked areas)
        """
        frame_length = len(frames)
        # Preprocess frames to tensors and normalize
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # Move feature tensor to specified device (CPU or GPU)
        feats = feats.to(self.device)
        # Initialize list of video length to store processed frames
        comp_frames = [None] * frame_length
        # Disable gradient calculation for inference efficiency
        with torch.no_grad():
            # Pass processed frames through encoder to generate feature representation
            feats = self.model.encoder(feats.view(frame_length, 3, self.model_input_height, self.model_input_width))
            # Get feature dimension info
            _, c, feat_h, feat_w = feats.size()
            # Reshape features to match model's expected input
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        # Get inpaint area
        # Iterate through video within set neighbor stride
        for f in range(0, frame_length, self.neighbor_stride):
            # Calculate neighbor IDs
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # Get reference frame indices
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            # Also disable gradient calculation
            with torch.no_grad():
                # Infer features through model and pass to decoder to generate completed frames
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                # Pass predicted features through decoder, apply tanh, and detach tensor
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                # Rescale result tensor to 0-255 (image pixel values)
                pred_img = (pred_img + 1) / 2
                # Move tensor back to CPU and convert to NumPy array
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # Iterate through neighbor frames
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # Convert predicted image to uint8 format
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        # If position is empty, assign newly calculated image
                        comp_frames[idx] = img
                    else:
                        # If image exists, mix old and new images to improve quality
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # Return processed frame sequence
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        """
        Get subtitle removal area, determine region and height based on mask
        """
        inpaint_area = []
        to_H = H
        while to_H > 0:
            from_H = max(0, to_H - h)
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                # Find where the contiguous mask block ends entirely
                move_down = 0
                while to_H + move_down < H and not np.all(mask[to_H + move_down, :] == 0):
                    move_down += 1
                
                move_up = 0
                while from_H - move_up > 0 and not np.all(mask[from_H - move_up - 1, :] == 0):
                    move_up += 1
                
                new_to_H = to_H + move_down
                new_from_H = from_H - move_up
                
                if new_to_H - new_from_H <= h:
                    diff = h - (new_to_H - new_from_H)
                    pad_down = min(diff, H - new_to_H)
                    new_to_H += pad_down
                    diff -= pad_down
                    new_from_H = max(0, new_from_H - diff)
                    
                    if (new_from_H, new_to_H) not in inpaint_area:
                        inpaint_area.append((new_from_H, new_to_H))
                    to_H = new_from_H
                else:
                    chunks = []
                    temp_to_H = new_to_H
                    while temp_to_H > new_from_H:
                        temp_from_H = max(new_from_H, temp_to_H - h)
                        if temp_to_H - temp_from_H < h:
                            diff = h - (temp_to_H - temp_from_H)
                            pad_down = min(diff, H - temp_to_H)
                            temp_to_H += pad_down
                            diff -= pad_down
                            temp_from_H = max(0, temp_from_H - diff)
                            
                        chunks.append((temp_from_H, temp_to_H))
                        temp_to_H = temp_from_H
                        
                    for c in chunks:
                        if c not in inpaint_area:
                            inpaint_area.append(c)
                    to_H = new_from_H
            else:
                to_H -= h
        return inpaint_area  # Return inpaint area list

    @staticmethod
    def get_inpaint_area_by_selection(input_sub_area, mask):
        print('use selection area for inpainting')
        height, width = mask.shape[:2]
        ymin, ymax, _, _ = input_sub_area
        interval_size = 135
        # List to store results
        inpaint_area = []
        # Calculate and store standard intervals
        for i in range(ymin, ymax, interval_size):
            inpaint_area.append((i, i + interval_size))
        # Check if last interval reaches maximum
        if inpaint_area[-1][1] != ymax:
            # If not, create new interval starting at end of last and ending at expanded value
            if inpaint_area[-1][1] + interval_size <= height:
                inpaint_area.append((inpaint_area[-1][1], inpaint_area[-1][1] + interval_size))
        return inpaint_area  # Return inpaint area list


class STTNVideoInpaint:

    def read_frame_info_from_video(self):
        # Read video using OpenCV
        reader = cv2.VideoCapture(self.video_path)
        # Get video width, height, fps, and frame count info and store in frame_info dict
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # Original width
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # Original height
            'fps': reader.get(cv2.CAP_PROP_FPS),  # FPS
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # Total frames
        }
        # Return video reader object, frame info, and video writer object
        return reader, frame_info

    def __init__(self, video_path, mask_path=None, clip_gap=None):
        # STTNInpaint video inpaint instance initialization
        self.sttn_inpaint = STTNInpaint()
        # Video and mask paths
        self.video_path = video_path
        self.mask_path = mask_path
        # Set output video file path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        # Configure max frames to load in one process
        if clip_gap is None:
            self.clip_gap = config.STTN_MAX_LOAD_NUM
        else:
            self.clip_gap = clip_gap

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader = None
        writer = None
        try:
            # Read video frame info
            reader, frame_info = self.read_frame_info_from_video()
            if input_sub_remover is not None:
                writer = input_sub_remover.video_writer
            else:
                # Create video writer object for output
                writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
            
            # Calculate number of iterations needed to inpaint video
            rec_time = frame_info['len'] // self.clip_gap if frame_info['len'] % self.clip_gap == 0 else frame_info['len'] // self.clip_gap + 1
            # Calculate split height for inpaint region size
            split_h = int(frame_info['W_ori'] * 3 / 16)
            
            if input_mask is None:
                # Read mask
                mask = self.sttn_inpaint.read_mask(self.mask_path)
            else:
                _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
                mask = mask[:, :, None]
                
            # Get inpaint area positions
            inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(frame_info['H_ori'], split_h, mask)
            if not inpaint_area:
                print('[Warning] Mask produced no inpaint areas — check that the mask covers the subtitle region. Frames will be written unchanged.')
            
            # Iterate through each segment
            for i in range(rec_time):
                start_f = i * self.clip_gap  # Start frame position
                end_f = min((i + 1) * self.clip_gap, frame_info['len'])  # End frame position
                print('Processing:', start_f + 1, '-', end_f, ' / Total:', frame_info['len'])
                
                frames_hr = []  # High resolution frames list
                frames = {}  # Frames dict for cropped images
                comps = {}  # Comps dict for inpainted images
                
                # Initialize frames dict
                for k in range(len(inpaint_area)):
                    frames[k] = []
                    
                # Read and inpaint high resolution frames
                valid_frames_count = 0
                for j in range(start_f, end_f):
                    success, image = reader.read()
                    if not success:
                        print(f"Warning: Failed to read frame {j}.")
                        break
                    
                    frames_hr.append(image)
                    valid_frames_count += 1
                    
                    for k in range(len(inpaint_area)):
                        # Crop, resize and add to frames dict
                        image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                        image_resize = cv2.resize(image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                        frames[k].append(image_resize)
                
                # If no valid frames read, skip current segment
                if valid_frames_count == 0:
                    print(f"Warning: No valid frames found in range {start_f+1}-{end_f}. Skipping this segment.")
                    continue
                    
                # Run inpaint for each region
                for k in range(len(inpaint_area)):
                    if len(frames[k]) > 0:  # Ensure frames are available to process
                        comps[k] = self.sttn_inpaint.inpaint(frames[k])
                    else:
                        comps[k] = []
                
                # If inpaint areas exist
                if inpaint_area and valid_frames_count > 0:
                    for j in range(valid_frames_count):
                        if input_sub_remover is not None and input_sub_remover.gui_mode:
                            original_frame = copy.deepcopy(frames_hr[j])
                        else:
                            original_frame = None
                            
                        frame = frames_hr[j]
                        
                        for k in range(len(inpaint_area)):
                            if j < len(comps[k]):  # Ensure index is valid
                                # Rescale inpainted image back to original resolution and fuse into original frame
                                custom_h = inpaint_area[k][1] - inpaint_area[k][0]
                                comp = cv2.resize(comps[k][j], (frame_info['W_ori'], custom_h))
                                comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                                mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                                frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                        
                        writer.write(frame)
                        
                        if input_sub_remover is not None:
                            if tbar is not None:
                                input_sub_remover.update_progress(tbar, increment=1)
                            if original_frame is not None and input_sub_remover.gui_mode:
                                input_sub_remover.preview_frame = cv2.hconcat([original_frame, frame])
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Only release the writer if WE created it (not borrowed from sub_remover)
            if writer and input_sub_remover is None:
                writer.release()


if __name__ == '__main__':
    mask_path = '../../test/test.png'
    video_path = '../../test/test.mp4'
    # Record start time
    start = time.time()
    sttn_video_inpaint = STTNVideoInpaint(video_path, mask_path, clip_gap=config.STTN_MAX_LOAD_NUM)
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')
