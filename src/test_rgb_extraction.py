print("TEST STARTED")

from image_processing import load_and_resize_image, extract_fingernail_roi, extract_color_features

def main():
    print("Inside main")
    image_path = 'test_images/sample.png'
    
    print("Loading image...")
    try:
        image = load_and_resize_image(image_path)
        print("Image loaded successfully.")
    except Exception as e:
        print("Error loading image:", e)
        return

    print("Extracting ROI...")
    roi = extract_fingernail_roi(image)
    print("ROI extracted.")

    print("Extracting RGB features...")
    rgb = extract_color_features(roi)

    r, g, b = rgb[2], rgb[1], rgb[0]
    print(f"Extracted Average RGB (R, G, B): ({r:.2f}, {g:.2f}, {b:.2f})")

if __name__ == "__main__":
    main()