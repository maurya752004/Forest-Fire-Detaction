from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def create_output_folders():
    # Create base output directory
    base_output_dir = Path("videooutput2")
    base_output_dir.mkdir(exist_ok=True)
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / timestamp
    output_dir.mkdir(exist_ok=True)
    
    return output_dir

def process_video(model_path, video_path, output_dir):
    # Load the trained model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    output_path = str(output_dir / f"processed_{Path(video_path).name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create a log file for predictions
    log_path = output_dir / "predictions.txt"
    with open(log_path, 'w') as f:
        f.write(f"Video Analysis Log: {Path(video_path).name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Resolution: {frame_width}x{frame_height}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write("\n=== Frame by Frame Analysis ===\n\n")
    
    frame_count = 0
    fire_frames = 0
    
    print(f"\nProcessing video: {video_path}")
    print(f"Total frames: {total_frames}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:  # Progress update every 10 frames
            print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
        
        # Enhanced preprocessing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        # Resize with better interpolation
        frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Make prediction without augmentation
        results = model.predict(frame_resized, imgsz=224)
        probs = results[0].probs.data.tolist()
        
        # Get prediction results
        probs = results[0].probs.data.tolist()
        predicted_class = results[0].probs.top1
        confidence = probs[predicted_class]
        class_name = model.names[predicted_class]
        
        # Boost fire detection sensitivity
        if class_name == "Fire":
            confidence = min(confidence * 1.2, 1.0)  # Boost fire confidence by 20% but cap at 100%
        
        # Optional: Further reduce non-fire confidence to increase fire sensitivity
        else:
            confidence = confidence * 0.9  # Reduce non-fire confidence by 10%
        
        # Add prediction text and box to frame
        text = f"{class_name}: {confidence:.2%}"
        if class_name == "Fire" and confidence > 0.2:  # Much lower threshold for better sensitivity
            color = (0, 0, 255)  # Red for fire
            fire_frames += 1
            
            # Add a red box around the frame to highlight fire detection
            frame_height, frame_width = frame.shape[:2]
            box_thickness = 3
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), color, box_thickness)
            
            # Save the frame with box
            frame_path = output_dir / f"fire_frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        else:
            color = (0, 255, 0)  # Green for non-fire
            
        # Add text overlay with background for better visibility
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.rectangle(frame, (10, 5), (10 + text_size[0], 35), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Log the prediction
        with open(log_path, 'a') as f:
            f.write(f"Frame {frame_count:04d}: {class_name} ({confidence:.2%})\n")
    
    # Write summary to log
    with open(log_path, 'a') as f:
        f.write(f"\n=== Analysis Summary ===\n")
        f.write(f"Total Frames Processed: {frame_count}\n")
        f.write(f"Frames with Fire Detected: {fire_frames}\n")
        f.write(f"Fire Detection Rate: {(fire_frames/frame_count)*100:.2f}%\n")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\nProcessing complete!")
    print(f"Output video saved to: {output_path}")
    print(f"Analysis log saved to: {log_path}")
    print(f"Fire frames saved in: {output_dir}")
    print("\nTo play the video, you can use:")
    print(f"vlc {output_path}  # If you have VLC installed")
    print(f"ffplay {output_path}  # If you have ffmpeg installed")

def main():
    # Create output folders
    output_dir = create_output_folders()
    print(f"Created output directory: {output_dir}")
    
    # Path to your trained model and video
    model_path = "fire_detection_yolo/yolo_fire_v1_cls2/weights/best.pt"
    video_path = "stream.mp4"
    
    # Process the video
    process_video(model_path, video_path, output_dir)

if __name__ == "__main__":
    # Create output directory
    output_dir = create_output_folders()
    
    # Path to your trained model and video
    model_path = "fire_detection_yolo/yolo_fire_v1_cls2/weights/best.pt"  # Using the YOLO model
    video_path = "stream.mp4"
    
    # Process the video
    process_video(model_path, video_path, output_dir)
    print("\nProcessing complete!")
    print(f"Output video saved to: {output_dir}/processed_{Path(video_path).name}")
    print(f"Analysis log saved to: {output_dir}/predictions.txt")
    print(f"Fire frames saved in: {output_dir}")
    
    print("\nTo play the video, you can use:")
    print("vlc video_output/latest/processed_stream.mp4  # If you have VLC installed")
    print("ffplay video_output/latest/processed_stream.mp4  # If you have ffmpeg installed")