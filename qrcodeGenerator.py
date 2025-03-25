import cv2
import numpy as np
import time
import os
import qrcode
from PIL import Image

def generate_qr_code(data, output_dir="qr_codes", unique_id=None):
    """Generate a QR code with just the animal name and save it as an image."""
    os.makedirs(output_dir, exist_ok=True)
    
    qr_data = data  # Use only the animal name
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data)
    qr.make(fit=True)
    
    img = qr.make_image(fill="black", back_color="white")
    img_path = os.path.join(output_dir, f"{qr_data}.png")
    img.save(img_path)
    print(f"QR Code saved: {img_path}")
    return img_path

def generate_aruco_marker(marker_id, marker_size=200, output_dir="markers"):
    """Generate an ArUco marker and save it as an image."""
    os.makedirs(output_dir, exist_ok=True)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    marker_path = os.path.join(output_dir, f"marker_{marker_id}.png")
    cv2.imwrite(marker_path, marker_img)
    print(f"ArUco marker saved: {marker_path}")
    return marker_path

def create_print_sheet(animals, outline_dir="outlines", output_file="print_sheet.png", sheet_size=(2480, 3508)):
    """Create a print sheet combining QR codes, outlines, and ArUco markers."""
    os.makedirs("print_sheets", exist_ok=True)
    
    for i, animal in enumerate(animals):
        sheet = Image.new("RGB", sheet_size, "white")
        qr_output_dir = "qr_codes"
        marker_output_dir = "markers"
        
        qr_size = 800  # Adjust QR code size
        outline_size = (2000, 2000)  # Resize outline images
        marker_size = 200  # ArUco marker size
        
        qr_x, qr_y = 50, 50  # QR code position in the top-left corner
        marker_positions = [  # Four corner markers (adjusted to avoid QR overlap)
            (qr_x + qr_size + 50, qr_y), (2280 - marker_size, 200),
            (200, 3308 - marker_size), (2280 - marker_size, 3308 - marker_size)
        ]
        
        qr_path = generate_qr_code(animal)
        outline_path = os.path.join(outline_dir, f"{animal}.png")
        
        if os.path.exists(outline_path):
            outline = Image.open(outline_path).convert("RGBA").resize(outline_size)
        else:
            outline = Image.new("RGBA", outline_size, "white")  # Placeholder if missing
        
        qr = Image.open(qr_path).resize((qr_size, qr_size))
        
        # Create a new sheet with the outline centered
        outline_x = (sheet_size[0] - outline_size[0]) // 2
        outline_y = (sheet_size[1] - outline_size[1]) // 2
        sheet.paste(outline, (outline_x, outline_y))
        
        # Place the QR code in the top-left corner
        sheet.paste(qr, (qr_x, qr_y))
        
        # Generate and place ArUco markers
        for j, pos in enumerate(marker_positions):
            marker_path = generate_aruco_marker(j)
            marker = Image.open(marker_path).convert("L").resize((marker_size, marker_size))
            sheet.paste(marker, pos)
        
        output_path = os.path.join("print_sheets", f"print_sheet_{animal}.png")
        sheet.save(output_path)
        print(f"Print sheet saved: {output_path}")

if __name__ == "__main__":
    animals = ["frog", "hedgehog", "lion", "tiger", "elephant"]  # Add more as needed
    create_print_sheet(animals)
