import os
import qrcode
from PIL import Image

def generate_qr_code(data, output_dir="qr_codes"):
    """Generate a QR code with just the animal name and save it as an image."""
    os.makedirs(output_dir, exist_ok=True)
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill="black", back_color="white")
    img_path = os.path.join(output_dir, f"{data}.png")
    img.save(img_path)
    print(f"QR Code saved: {img_path}")
    return img_path

def generate_aruco_marker(marker_id, marker_size=200, marker_dir="markers"):
    """Load a pre-generated ArUco marker from disk."""
    marker_path = os.path.join(marker_dir, f"marker_{marker_id}.png")
    if not os.path.exists(marker_path):
        raise FileNotFoundError(f"Missing ArUco marker image: {marker_path}")
    
    marker = Image.open(marker_path).convert("L").resize((marker_size, marker_size))
    return marker

def create_print_sheet(animals, outline_dir="outlines", output_dir="print_sheets", sheet_size=(2480, 3508)):
    """Create print sheets with QR codes, outlines, and ArUco markers."""
    os.makedirs(output_dir, exist_ok=True)
    
    for animal in animals:
        print(f"\nCreating sheet for: {animal}")
        sheet = Image.new("RGB", sheet_size, "white")

        qr_size = 800
        outline_size = (2000, 2000)
        marker_size = 200

        qr_x = (sheet_size[0] - qr_size) // 2
        qr_y = 50

        outline_x = (sheet_size[0] - outline_size[0]) // 2
        outline_y = (sheet_size[1] - outline_size[1]) // 2

        marker_positions = [
            (0, 0),  # Top-left (ID 0)
            (sheet_size[0] - marker_size, 0),  # Top-right (ID 1)
            (0, sheet_size[1] - marker_size),  # Bottom-left (ID 2)
            (sheet_size[0] - marker_size, sheet_size[1] - marker_size),  # Bottom-right (ID 3)
        ]

        # Add outline
        outline_path = os.path.join(outline_dir, f"{animal}.png")
        if os.path.exists(outline_path):
            outline = Image.open(outline_path).convert("RGBA").resize(outline_size)
        else:
            outline = Image.new("RGBA", outline_size, "white")  # Blank placeholder
        sheet.paste(outline, (outline_x, outline_y), outline)

        # Add QR code
        qr_path = generate_qr_code(animal)
        qr = Image.open(qr_path).resize((qr_size, qr_size))
        sheet.paste(qr, (qr_x, qr_y))

        # Add ArUco markers
        marker_ids = [0, 1, 2, 3]
        for j, pos in enumerate(marker_positions):
            marker_img = generate_aruco_marker(marker_ids[j], marker_size)
            sheet.paste(marker_img, pos)

        # Save final sheet
        output_path = os.path.join(output_dir, f"print_sheet_{animal}.png")
        sheet.save(output_path)
        print(f"Print sheet saved: {output_path}")

if __name__ == "__main__":
    animals = ["frog", "hedgehog", "lion", "tiger", "elephant"]  # Add more if needed
    create_print_sheet(animals)
