import cv2
import sys
import json
import numpy as np
from ultralytics import YOLO
import os

def main():
    if len(sys.argv) < 3: 
        print("Error: Missing arguments", flush=True)
        return
        
    video_path = sys.argv[1]
    output_path = sys.argv[2] 
    
    # Load Model
    model = YOLO("yolov8n") 
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get Original Dimensions
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    full_match_data = []
    frame_idx = 0
    
    print("PROGRESS:1", flush=True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # Update progress every 10 frames (Less I/O overhead)
        if frame_idx % 10 == 0:
            # Check division by zero if total_frames is 0
            if total_frames > 0:
                progress = int((frame_idx / total_frames) * 100)
                print(f"PROGRESS:{progress}", flush=True)

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # 2. TRACKING
        # imgsz=640: Downscales internally for speed (faster than 1024, usually good enough for demo)
        # conf=0.25: Standard confidence. 0.15 might give too many "ghost" balls.
        results = model.track(
            frame, 
            persist=True, 
            conf=0.25, 
            imgsz=640, 
            tracker="bytetrack.yaml", 
            classes=[0, 32], 
            verbose=False
        )
        
        frame_data = {'t': round(timestamp, 3), 'p': [], 'b': None}

        for r in results:
            if r.boxes is None: continue
            
            ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None]*len(r.boxes)
            clss = r.boxes.cls.int().cpu().tolist()
            coords = r.boxes.xyxy.cpu().tolist()

            for box_id, cls, bbox in zip(ids, clss, coords):
                # Normalize coordinates using ORIGINAL dimensions
                # This ensures the boxes align perfectly on the frontend regardless of resizing
                nx = round(bbox[0] / orig_w, 4)
                ny = round(bbox[1] / orig_h, 4)
                nw = round((bbox[2] - bbox[0]) / orig_w, 4)
                nh = round((bbox[3] - bbox[1]) / orig_h, 4)
                
                if cls == 32: # Ball
                    frame_data['b'] = [nx, ny, nw, nh]
                elif cls == 0 and box_id is not None: # Player
                    # Simple Team Logic (Left/Right split)
                    # Note: This is a heuristic for the demo. Real team ID requires color clustering.
                    team = 0 if (nx + nw/2) < 0.5 else 1
                    frame_data['p'].append([box_id, nx, ny, nw, nh, team])

        full_match_data.append(frame_data)
        frame_idx += 1

    cap.release()
    
    # Write JSON
    try:
        with open(output_path, 'w') as f: 
            json.dump(full_match_data, f)
        print("PROGRESS:100", flush=True)
        print("DONE", flush=True)
    except Exception as e:
        print(f"ERROR:Failed to write JSON - {e}", flush=True)

if __name__ == "__main__":
    main()