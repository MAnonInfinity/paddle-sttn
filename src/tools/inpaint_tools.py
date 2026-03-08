import multiprocessing
import cv2
import numpy as np

from src import config
from src.inpaint.lama_inpaint import LamaInpaint


def batch_generator(data, max_batch_size):
    """
    Generate uniform batch data with max length not exceeding max_batch_size based on data size
    """
    n_samples = len(data)
    # Try to find a batch_size smaller than MAX_BATCH_SIZE to make batch sizes as close as possible
    batch_size = max_batch_size
    num_batches = n_samples // batch_size

    # Handle case where last batch is smaller than batch_size
    # If last batch is smaller than others, decrease batch_size to balance
    while n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1  # Decrease batch size
        num_batches = n_samples // batch_size

    # Generate first num_batches
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    # Use remaining data as last batch
    last_batch_start = num_batches * batch_size
    if last_batch_start < n_samples:
        yield data[last_batch_start:]


def inference_task(batch_data):
    inpainted_frame_dict = dict()
    for data in batch_data:
        index, original_frame, coords_list = data
        mask_size = original_frame.shape[:2]
        mask = create_mask(mask_size, coords_list)
        inpaint_frame = inpaint(original_frame, mask)
        inpainted_frame_dict[index] = inpaint_frame
    return inpainted_frame_dict


def parallel_inference(inputs, batch_size=None, pool_size=None):
    """
    Parallel inference while maintaining result order
    """
    if pool_size is None:
        pool_size = multiprocessing.cpu_count()
    # Use context manager to manage process pool
    with multiprocessing.Pool(processes=pool_size) as pool:
        batched_inputs = list(batch_generator(inputs, batch_size))
        # Use map to ensure input and output order matches
        batch_results = pool.map(inference_task, batched_inputs)
    # Flatten batch results
    index_inpainted_frames = [item for sublist in batch_results for item in sublist]
    return index_inpainted_frames


def inpaint(img, mask):
    lama_inpaint_instance = LamaInpaint()
    img_inpainted = lama_inpaint_instance(img, mask)
    return img_inpainted


def inpaint_with_multiple_masks(censored_img, mask_list):
    inpainted_frame = censored_img
    if mask_list:
        for mask in mask_list:
            inpainted_frame = inpaint(inpainted_frame, mask)
    return inpainted_frame


def create_mask(size, coords_list):
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            # Expand by pixels to avoid tight boxes
            x1 = xmin - config.SUBTITLE_AREA_DEVIATION_PIXEL
            if x1 < 0:
                x1 = 0
            y1 = ymin - config.SUBTITLE_AREA_DEVIATION_PIXEL
            if y1 < 0:
                y1 = 0
            x2 = xmax + config.SUBTITLE_AREA_DEVIATION_PIXEL
            y2 = ymax + config.SUBTITLE_AREA_DEVIATION_PIXEL
            cv2.rectangle(mask, (x1, y1),
                          (x2, y2), 255, thickness=-1)
    return mask


def inpaint_video(video_path, sub_list):
    index = 0
    frame_to_inpaint_list = []
    video_cap = cv2.VideoCapture(video_path)
    while True:
        # Read video frame
        ret, frame = video_cap.read()
        if not ret:
            break
        index += 1
        if index in sub_list.keys():
            frame_to_inpaint_list.append((index, frame, sub_list[index]))
        if len(frame_to_inpaint_list) > config.PROPAINTER_MAX_LOAD_NUM:
            batch_results = parallel_inference(frame_to_inpaint_list)
            for index, frame in batch_results:
                temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'test', 'temp')
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                file_name = os.path.join(temp_dir, f'{index}.png')
                cv2.imwrite(file_name, frame)
                print(f"success write: {file_name}")
            frame_to_inpaint_list.clear()
    print(f'finished')


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
