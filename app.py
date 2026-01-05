import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

model = YOLO("best.pt")

# Check what the actual class names are
print("Model class names:", model.names)

def process_image(img):
    """Process single image for basketball player detection"""
    if img is None:
        return None, "Please upload an image"
    
    try:
        # Convert PIL to BGR for OpenCV
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model(img_array, verbose=False)
        
        player_count = 0
        
        # Draw boxes
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Orange/Basketball color for players
            color = (0, 140, 255)  # Orange in BGR
            label = f"Player #{i+1}: {confidence:.2f}"
            
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_array, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            player_count += 1
        
        # Convert back to RGB for display
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Summary
        if player_count > 0:
            info = f"🏀 Basketball Player Detection\n"
            info += f"👥 Players detected: {player_count}\n"
            info += f"✅ Detection complete!"
        else:
            info = "No basketball players detected"
        
        return img_rgb, info
    
    except Exception as e:
        return None, f"Error: {e}"

def process_video(video_path):
    """Process video for basketball player detection"""
    if video_path is None:
        return None, "Please upload a video"
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "Error: Could not open video file"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video path
        output_path = "output_basketball_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        max_players_in_frame = 0
        min_players_in_frame = float('inf')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, verbose=False)
            
            players_in_frame = len(results[0].boxes)
            total_detections += players_in_frame
            max_players_in_frame = max(max_players_in_frame, players_in_frame)
            if players_in_frame > 0:
                min_players_in_frame = min(min_players_in_frame, players_in_frame)
            
            # Draw boxes
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Orange color for basketball players
                color = (0, 140, 255)  # Orange in BGR
                label = f"Player {i+1}: {confidence:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add statistics overlay
            cv2.putText(frame, f"Frame: {frame_count+1}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Players on court: {players_in_frame}", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        if not os.path.exists(output_path):
            return None, "Error: Video output file was not created"
        
        # Summary
        avg_players = total_detections / total_frames if total_frames > 0 else 0
        if min_players_in_frame == float('inf'):
            min_players_in_frame = 0
        
        info = f"✅ Video processed!\n"
        info += f"🏀 Basketball Game Statistics:\n"
        info += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        info += f"📹 Total frames: {total_frames}\n"
        info += f"👥 Total player detections: {total_detections}\n"
        info += f"📊 Average players per frame: {avg_players:.1f}\n"
        info += f"📈 Max players in frame: {max_players_in_frame}\n"
        info += f"📉 Min players in frame: {min_players_in_frame}\n"
        
        return output_path, info
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg

# Create Gradio interface with tabs
with gr.Blocks(title="Basketball Player Detection") as demo:
    gr.Markdown("# 🏀 Basketball Player Detection System")
    gr.Markdown("Detect and track basketball players in game footage")
    
    with gr.Tab("📷 Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Basketball Image")
                image_button = gr.Button("🔍 Detect Players", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Detection Result")
                image_info = gr.Textbox(label="Player Analysis", lines=4)
        
        image_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[image_output, image_info]
        )
        
        gr.Markdown("**🟠 Orange Boxes = Basketball Players**")
    
    with gr.Tab("🎥 Video Detection"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Basketball Game Video")
                video_button = gr.Button("🎬 Process Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Processed Video with Detections")
                video_info = gr.Textbox(label="Game Statistics", lines=10)
        
        video_button.click(
            fn=process_video,
            inputs=video_input,
            outputs=[video_output, video_info]
        )
        
        gr.Markdown("**Note:** Video processing may take a few minutes. Larger videos will take longer.")

if __name__ == "__main__":
    demo.launch()