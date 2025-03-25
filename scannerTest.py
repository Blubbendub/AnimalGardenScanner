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

    return None, corners, ids

def align_with_aruco_markers(image, marker_corners, ids, sheet_size):
    """Align image based on the four ArUco markers (IDs 0–3)."""
    if marker_corners is None or ids is None or len(marker_corners) < 4:
        print("Error: Not enough ArUco markers detected.")
        return None

    # Expected locations on the print sheet
    id_to_position = {
        0: [0, 0],
        1: [sheet_size[0] - 1, 0],
        2: [0, sheet_size[1] - 1],
        3: [sheet_size[0] - 1, sheet_size[1] - 1]
    }

    actual_points = []
    expected_points = []

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in id_to_position:
            actual_points.append(marker_corners[i][0])
            expected_points.append(id_to_position[marker_id])

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

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = image
    result[:, :, 3] = binary_mask

    return result

def detect_qr_and_process(frame):
    """Main logic to detect and extract the drawing region using ArUco markers."""
    qr_data, marker_corners, ids = detect_markers(frame)

    if qr_data is None:
        print("No QR code found.")
        return frame

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

    # Drawing area (outline) is centered and 2000x2000 on the sheet
    outline_x = (sheet_size[0] - 2000) // 2
    outline_y = (sheet_size[1] - 2000) // 2
    cropped = aligned[outline_y:outline_y + 2000, outline_x:outline_x + 2000]

    # Resize to 1000x1000 to match mask resolution
    cropped_resized = cv2.resize(cropped, (1000, 1000))
    mask_resized = cv2.resize(mask, (1000, 1000))

    result = apply_bw_mask(cropped_resized, mask_resized)

    output_path = f"output_{qr_data}.png"
    cv2.imwrite(output_path, result, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"Saved: {output_path}")

    # Optional debug
    debug = aligned.copy()
    cv2.rectangle(debug, (outline_x, outline_y),
                  (outline_x + 2000, outline_y + 2000), (0, 0, 255), 5)
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
            # Right after processed_frame is computed:
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
