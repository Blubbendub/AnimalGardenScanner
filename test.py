import cv2

print("OpenCV:", cv2.__version__)
print("Has aruco:", hasattr(cv2, "aruco"))
print("Has ArucoDetector:", hasattr(cv2.aruco, "ArucoDetector"))

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        print("✅ Detected ArUco IDs:", ids.flatten())
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    else:
        print("❌ No markers detected.")

    cv2.imshow("Aruco Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()