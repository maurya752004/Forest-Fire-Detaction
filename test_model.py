from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def create_output_folders():
    # Create base output directory
    base_output_dir = Path("output")
    base_output_dir.mkdir(exist_ok=True)
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for fire and non-fire
    fire_dir = output_dir / "fire"
    non_fire_dir = output_dir / "non_fire"
    fire_dir.mkdir(exist_ok=True)
    non_fire_dir.mkdir(exist_ok=True)
    
    return output_dir

def test_image(model_path, image_path, output_dir):
    # Load the trained model
    model = YOLO(model_path)
    
    # Read and process the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Perform prediction
    results = model.predict(image, imgsz=224)
    
    # Get the prediction results
    probs = results[0].probs.data.tolist()  # Probability for each class
    predicted_class = results[0].probs.top1  # Index of highest probability
    confidence = probs[predicted_class]  # Confidence score
    class_name = model.names[predicted_class]  # Class name
    
    # Display results
    print(f"\nResults for {image_path}:")
    print(f"Predicted Class: {class_name}")
    print(f"Confidence: {confidence:.2%}")
    
    # Draw on image
    img_with_text = image.copy()
    text = f"{class_name}: {confidence:.2%}"
    cv2.putText(img_with_text, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Determine which subfolder to use based on prediction
    class_folder = output_dir / ('fire' if class_name == 'Fire' else 'non_fire')
    
    # Save the original image
    orig_output_path = str(class_folder / f"orig_{Path(image_path).name}")
    cv2.imwrite(orig_output_path, image)
    
    # Save the annotated image
    output_path = str(class_folder / f"pred_{Path(image_path).name}")
    cv2.imwrite(output_path, img_with_text)
    
    # Save prediction info in a text file
    info_path = str(class_folder / f"pred_{Path(image_path).stem}.txt")
    with open(info_path, 'w') as f:
        f.write(f"Image: {Path(image_path).name}\n")
        f.write(f"Prediction: {class_name}\n")
        f.write(f"Confidence: {confidence:.2%}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Original image saved to: {orig_output_path}")
    print(f"Annotated image saved to: {output_path}")
    print(f"Prediction info saved to: {info_path}\n")

def main():
    # Create output folders
    output_dir = create_output_folders()
    print(f"Created output directory: {output_dir}")
    
    # Path to your trained model
    model_path = "fire_detection_yolo/yolo_fire_v1_cls2/weights/best.pt"
    
    # Test images from validation set
    test_images = [
        "dataset/images/val/Fire/F_1.jpg",
        "dataset/images/val/Non_Fire/NF_1.jpg"
    ]
    
    # Process each image
    print("\nProcessing images...")
    for image_path in test_images:
        print(f"\nProcessing: {image_path}")
        test_image(model_path, image_path, output_dir)
    
    print(f"\nAll results saved in: {output_dir}")

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()  # Clean up all windows at the end