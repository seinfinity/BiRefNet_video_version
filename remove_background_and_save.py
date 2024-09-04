import os
import cv2
import numpy as np

# Directories containing frames and masks
input_frames_dir = ''
mask_frames_dir = ''
output_frames_dir = ''

# Ensure the output directory exists
os.makedirs(output_frames_dir, exist_ok=True)

# List all frame and mask files (assuming they are sorted numerically/alphabetically)
input_frames = sorted([f for f in os.listdir(input_frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
mask_frames = sorted([f for f in os.listdir(mask_frames_dir) if f.endswith('.jpg') or f.endswith('.png')])

# Verify that the number of frames and masks are equal
if len(input_frames) != len(mask_frames):
    print("Error: The number of input frames and mask frames must be the same.")
    exit()

print(f"Processing {len(input_frames)} frames...")

# Process each frame
for i in range(len(input_frames)):
    input_frame_path = os.path.join(input_frames_dir, input_frames[i])
    mask_frame_path = os.path.join(mask_frames_dir, mask_frames[i])

    # Read the input frame and corresponding mask frame
    frame = cv2.imread(input_frame_path)
    mask_image = cv2.imread(mask_frame_path, cv2.IMREAD_GRAYSCALE)

    if frame is None or mask_image is None:
        print(f"Error reading frame or mask at index {i}")
        continue

    # Ensure the data type of white background and result image are the same
    white_background = np.ones_like(frame, dtype=frame.dtype) * 255  # White background with the same dtype as frame

    # No Gaussian blur applied; use the original mask
    mask_normalized = mask_image / 255.0

    # Convert mask to 3 channels
    mask_3channel = cv2.merge([mask_normalized, mask_normalized, mask_normalized])

    # Perform alpha blending with the mask and white background
    result_on_white_bg = (frame * mask_3channel + white_background * (1 - mask_3channel)).astype(np.uint8)

    # Generate output frame path
    output_frame_path = os.path.join(output_frames_dir, f'frame_{i:03d}.png')

    # Save the processed frame to the output folder
    cv2.imwrite(output_frame_path, result_on_white_bg)
    print(f"Saved processed frame to {output_frame_path}")

print("All frames processed and saved successfully.")
