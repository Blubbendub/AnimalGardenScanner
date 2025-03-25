import cv2
import numpy as np
import time
import os

def detect_markers(frame):
    """Detect QR code for data, and ArUco markers for alignment."""
    # Detect QR code
    qr_detector = cv2.QRCodeDetector()
    qr_data, points, _ = qr_detector.detectAndDecode(frame)

    if points is not None and points.shape[1] == 4:
        frame = cv2.polylines(frame, [np.int32(points)], True, (0, 255, 0), 2)

    # Detect ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = aruco_detector.detectMarkers(frame)

    if corners:
        for corner in corners:
            frame = cv2.polylines(frame, [np.int32(corner)], True, (255, 0, 0), 2)

    if ids is not None:
        print(f"Detected ArUco IDs: {ids.flatten()}")
    else:
        print("No ArUco markers detected.")
    if qr_data.strip() != "":
        print(f"QR Code data: {qr_data}")
    
    return qr_data.strip(), corners, ids

def align_with_aruco_markers(image, marker_corners, ids, sheet_size):
    """Align image based on the four ArUco markers (IDs 0–3)."""
    if marker_corners is None or ids is None or len(marker_corners) < 4:
        print("Error: Not enough ArUco markers detected.")
        return None

    # Expected marker centers based on the print sheet design:
    # Markers were placed with a margin=50 and marker_size=500, so centers are:
    # Marker 0: (50+250, 50+250) = (300, 300)
    # Marker 1: (sheet_width - 50 - 250, 300) = (2180, 300)
    # Marker 2: (300, sheet_height - 50 - 250) = (300, 3208)
    # Marker 3: (2180, 3208)
    expected_centers = {
        0: [300, 300],
        1: [2180, 300],
        2: [300, 3208],
        3: [2180, 3208]
    }

    actual_points = []
    expected_points = []

    # Use the center of each detected marker for alignment
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in expected_centers:
            center = np.mean(marker_corners[i][0], axis=0)
            actual_points.append(center)
            expected_points.append(expected_centers[marker_id])

    if len(actual_points) != 4:
        print("Error: Missing some required marker IDs (0–3).")
        return None

    actual_points = np.array(actual_points, dtype='float32')
    expected_points = np.array(expected_points, dtype='float32')

    M, _ = cv2.findHomography(actual_points, expected_points, cv2.RANSAC, 5.0)
    if M is None:
        print("Homography computation failed.")
        return None

    aligned = cv2.warpPerspective(image, M, sheet_size)
    return aligned

def apply_bw_mask(image, mask):
    """Apply a black and white mask to the image where white areas are kept."""
    if mask is None:
        print("Error: Mask image not found.")
        return image

    # The mask is assumed to be the same size as the image.
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    # Create a 4-channel result image with an alpha channel from the mask
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = image
    result[:, :, 3] = binary_mask

    return result

def detect_qr_and_process(frame):
    """Main logic to detect and extract the drawing region using ArUco markers, then apply the mask."""
    qr_data, marker_corners, ids = detect_markers(frame)

    if qr_data is None or qr_data == "":
        print("No QR code found.")
        return frame

    # Load mask based on QR code data (e.g., animal name)
    mask_path = os.path.join("masks", f"{qr_data}.png")
    if not os.path.exists(mask_path):
        print(f"Error: Mask for '{qr_data}' not found.")
        return frame

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("Error: Could not load mask image.")
        return frame

    sheet_size = (2480, 3508)
    aligned = align_with_aruco_markers(frame, marker_corners, ids, sheet_size)

    if aligned is None:
        print("Could not align image with markers.")
        return frame

    # Drawing area (outline) is centered and originally 2000x2000 on the sheet.
    outline_x = (sheet_size[0] - 2000) // 2
    outline_y = (sheet_size[1] - 2000) // 2

    # Increase the region by a margin (delta) to cover a slightly larger area.
    delta = 50
    # Apply a slight vertical adjustment (if needed)
    vertical_adjustment = -25

    cropped = aligned[outline_y - delta + vertical_adjustment : outline_y + 2000 + delta + vertical_adjustment,
                      outline_x - delta : outline_x + 2000 + delta]

    # Resize cropped area to 1000x1000.
    cropped_resized = cv2.resize(cropped, (1000, 1000))
    
    # Instead of resizing the mask to 1000x1000 directly, create a slightly smaller mask.
    target_mask_size = 950  # New, smaller size for the mask content.
    mask_small = cv2.resize(mask, (target_mask_size, target_mask_size))
    # Create a blank mask of 1000x1000.
    if len(mask.shape) == 3:
        blank_mask = np.zeros((1000, 1000, mask.shape[2]), dtype=mask.dtype)
    else:
        blank_mask = np.zeros((1000, 1000), dtype=mask.dtype)
    start_x = (1000 - target_mask_size) // 2
    start_y = (1000 - target_mask_size) // 2
    blank_mask[start_y:start_y+target_mask_size, start_x:start_x+target_mask_size] = mask_small
    mask_final = blank_mask

    # Apply the mask to the cropped drawing area.
    result = apply_bw_mask(cropped_resized, mask_final)

    output_path = f"output_{qr_data}.png"
    cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Saved: {output_path}")

    # Debug image: draw the adjusted red rectangle
    debug = aligned.copy()
    cv2.rectangle(debug, (outline_x - delta, outline_y - delta + vertical_adjustment),
                  (outline_x + 2000 + delta, outline_y + 2000 + delta + vertical_adjustment), (0, 0, 255), 5)
    cv2.imwrite("debug_alignment_outline.png", debug)

    return result

def main():
    cap = cv2.VideoCapture(0)
    last_capture_time = 0
    capture_interval = 2
    processing_delay = 1

    while True:
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            time.sleep(processing_delay)
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't capture frame.")
                continue

            processed_frame = detect_qr_and_process(frame)
            cv2.imshow("Processed Image", processed_frame)
            last_capture_time = current_time

            # Show debug info for ArUco detection
            vis_frame = frame.copy()
            qr_data, corners, ids = detect_markers(vis_frame)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
            cv2.imshow("ArUco Detection Debug", vis_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
