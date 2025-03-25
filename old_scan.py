import cv2
import numpy as np
import time
import os

def detect_markers(frame):
    """Detect QR code and additional markers for better alignment."""
    detector = cv2.QRCodeDetector()
    qr_data, points, _ = detector.detectAndDecode(frame)
    
    # Detect ArUco markers for additional alignment
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    
    if qr_data and points is not None and ids is not None:
        print(f"QR Code detected: {qr_data}. Additional markers found: {len(ids)}")
        return qr_data, points[0].astype(np.float32), corners, ids
    
    print("No QR Code or additional markers detected.")
    return None, None, None, None

def align_image_with_markers(image, mask, qr_corners, marker_corners, sheet_size, outline_position, outline_size):
    """Align the captured image using QR code and additional markers."""
    if qr_corners is None or len(qr_corners) < 4:
        print("Error: QR code corners not detected correctly.")
        return image
    
    # Define the expected QR code corners in the print sheet layout
    expected_qr_corners = np.array([
        [50, 50], [850, 50], [850, 850], [50, 850]  # Matches QR position and size in the print layout
    ], dtype='float32')
    
    if marker_corners is not None and len(marker_corners) >= 4:
        # Use ArUco markers for more accurate alignment
        print("Using additional markers for refinement.")
        expected_marker_corners = np.array([
            [outline_position[0], outline_position[1]],
            [outline_position[0] + outline_size[0], outline_position[1]],
            [outline_position[0] + outline_size[0], outline_position[1] + outline_size[1]],
            [outline_position[0], outline_position[1] + outline_size[1]]
        ], dtype='float32')
        
        all_corners = np.vstack((qr_corners, marker_corners[:4].reshape(-1, 2)))
        expected_all_corners = np.vstack((expected_qr_corners, expected_marker_corners))
        
        M, _ = cv2.findHomography(all_corners, expected_all_corners, cv2.RANSAC)
    else:
        M = cv2.getPerspectiveTransform(qr_corners, expected_qr_corners)
    
    aligned_img = cv2.warpPerspective(image, M, sheet_size)
    
    # Extract the outline region
    outline_x, outline_y = outline_position
    cropped_img = aligned_img[outline_y:outline_y + outline_size[1], outline_x:outline_x + outline_size[0]]
    
    # Resize to match the final output size (600x600)
    final_output = cv2.resize(cropped_img, (600, 600))
    
    return final_output

def apply_bw_mask(image, mask):
    """Apply a black and white mask to the image where white areas are kept."""
    if mask is None:
        print("Error: Mask image not found.")
        return image
    
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # Convert mask to binary format
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if len(mask.shape) == 3 else mask
    _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Create a transparent output
    result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    result[:, :, :3] = image
    result[:, :, 3] = binary_mask
    
    return result

def detect_qr_and_process(frame):
    """Detect QR code and additional markers for improved alignment."""
    qr_data, qr_corners, marker_corners, ids = detect_markers(frame)
    
    if qr_data and qr_corners is not None:
        mask_path = os.path.join("masks", f"{qr_data}.png")
        
        if not os.path.exists(mask_path):
            print(f"Error: Mask for '{qr_data}' not found.")
            return frame
        
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None:
            print("Error: Mask image could not be loaded.")
            return frame
        
        # Define print sheet parameters
        sheet_size = (2480, 3508)  # A4 print dimensions
        outline_position = ((sheet_size[0] - 2000) // 2, (sheet_size[1] - 2000) // 2)  # Centered outline
        outline_size = (2000, 2000)  # Size of the outline area on print sheet
        
        aligned_frame = align_image_with_markers(frame, mask, qr_corners, marker_corners, sheet_size, outline_position, outline_size)
        masked_frame = apply_bw_mask(aligned_frame, mask)
        
        # Save the processed image
        output_filename = f"output_{qr_data}.png"
        cv2.imwrite(output_filename, masked_frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"Processed image saved: {output_filename}")
        
        return masked_frame
    
    print("No QR Code detected.")
    return frame  # Return original frame if no QR code is found

def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    last_capture_time = 0
    capture_interval = 2  # Capture every 2 seconds
    processing_delay = 1  # Wait 1 second before processing
    
    while True:
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            time.sleep(processing_delay)  # Wait before taking the image
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't capture frame.")
                continue
            
            processed_frame = detect_qr_and_process(frame)
            
            cv2.imshow("Processed Image", processed_frame)
            last_capture_time = current_time  # Update last capture time
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
