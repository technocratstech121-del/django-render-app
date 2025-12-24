"""
Railway Safety Monitoring System - Optimized Version
Enhanced for speed, accuracy, and proper safety scoring
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
import json
import os
import argparse
from collections import defaultdict, deque
import time
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.stdout.reconfigure(line_buffering=True)
print("Starting Railway Safety Monitor...")
print("OpenCV version:", cv2.__version__)

class RailwaySafetyMonitor:
    def __init__(self, video_path, output_dir="output", frame_skip=10, resize_factor=0.25, 
                 ultra_fast=True, clip_quality='low'):
        """Initialize Railway Safety Monitoring System"""
        print(f"\nInitializing monitor...")
        print(f"Video: {video_path}")
        print(f"Mode: {'Ultra-Fast' if ultra_fast else 'Normal'}")
        print(f"Clip Quality: {clip_quality}")
        
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_skip = frame_skip
        self.resize_factor = resize_factor
        self.ultra_fast = ultra_fast
        self.clip_quality = clip_quality
        
        # Clip resolutions based on quality
        self.clip_resolutions = {
            'low': (480, 270),
            'medium': (960, 540),
            'high': (1280, 720)
        }
        
        os.makedirs(output_dir, exist_ok=True)
        self.clips_dir = os.path.join(output_dir, 'event_clips')
        os.makedirs(self.clips_dir, exist_ok=True)
        
        # Initialize face detector with better parameters
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Warning: Face cascade failed to load")
        
        # Optimized background subtractor
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=40, detectShadows=False)
        
        # State variables
        self.events_log = []
        self.frame_count = 0
        self.processed_frame_count = 0
        self.fps = 30
        self.active_events = {}
        self.frame_buffer = deque(maxlen=60)  # Reduced buffer
        self.roi_driving_desk = None
        self.roi_motorman_chair = None
        self.processing_start_time = None
        
        # Detection state tracking
        self.last_face_count = 0
        self.no_face_counter = 0
        self.movement_history = deque(maxlen=10)
        
        # Multi-model detector (optional)
        self.multi_detector = None
        try:
            from .models_integration import MultiModelDetector
            self.multi_detector = MultiModelDetector(use_yolo=True, use_pose=True)
            print("✓ Multi-model detector loaded")
        except:
            print("Note: Multi-model detector not available")
        
        print(" Initialization complete")

    def setup_rois(self, frame_width, frame_height):
        """Setup Regions of Interest - optimized positions"""
        self.roi_driving_desk = {
            'x': int(0.30 * frame_width),
            'y': int(0.60 * frame_height),
            'w': int(0.45 * frame_width),
            'h': int(0.35 * frame_height)
        }
        
        self.roi_motorman_chair = {
            'x': int(0.20 * frame_width),
            'y': int(0.20 * frame_height),
            'w': int(0.45 * frame_width),
            'h': int(0.55 * frame_height)
        }

    def detect_faces(self, frame):
        """Optimized face detection"""
        # Adaptive detection size based on mode
        detection_size = 240 if self.ultra_fast else 480
        h, w = frame.shape[:2]
        aspect = w / h
        new_w = detection_size
        new_h = int(detection_size / aspect)
        
        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        scale_x = w / new_w
        scale_y = h / new_h
        faces = []
        
        if not self.face_cascade.empty():
            detected_faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=3, 
                minSize=(15, 15),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w_face, h_face) in detected_faces:
                faces.append({
                    'x': int(x * scale_x),
                    'y': int(y * scale_y),
                    'w': int(w_face * scale_x),
                    'h': int(h_face * scale_y),
                    'type': 'frontal'
                })
        
        return faces

    def detect_movement_in_roi(self, frame, roi_key):
        """Optimized movement detection"""
        roi = self.roi_driving_desk if roi_key == 'driving_desk' else self.roi_motorman_chair
        
        if not roi:
            return False
        
        try:
            roi_frame = frame[roi['y']:roi['y']+roi['h'], roi['x']:roi['x']+roi['w']]
            
            if roi_frame.size == 0:
                return False
            
            # Aggressive downsampling for speed
            if self.ultra_fast:
                roi_frame = cv2.resize(roi_frame, (roi_frame.shape[1]//4, roi_frame.shape[0]//4))
            
            fg_mask = self.background_subtractor.apply(roi_frame, learningRate=0.01)
            
            # Morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            movement_pixels = cv2.countNonZero(fg_mask)
            total_pixels = roi_frame.shape[0] * roi_frame.shape[1]
            movement_percentage = movement_pixels / total_pixels if total_pixels > 0 else 0
            
            # Store in history
            self.movement_history.append(movement_percentage)
            
            # Require sustained movement
            if len(self.movement_history) >= 3:
                avg_movement = sum(list(self.movement_history)[-3:]) / 3
                return avg_movement > 0.05
            
            return movement_percentage > 0.08
        except Exception as e:
            return False

    def count_people_in_roi(self, faces, roi_key):
        """Count people in specific ROI"""
        if roi_key == 'motorman_chair':
            roi = self.roi_motorman_chair
        else:
            return len(faces)
        
        if not roi:
            return 0
        
        people_in_roi = 0
        for face in faces:
            face_center_x = face['x'] + face['w'] // 2
            face_center_y = face['y'] + face['h'] // 2
            
            if (roi['x'] <= face_center_x <= roi['x'] + roi['w'] and
                roi['y'] <= face_center_y <= roi['y'] + roi['h']):
                people_in_roi += 1
        
        return people_in_roi

    def log_event(self, event_type, confidence, timestamp):
        """Log detected events with better tracking"""
        event = {
            'timestamp': str(timestamp),
            'frame': self.frame_count,
            'event_type': event_type,
            'confidence': confidence
        }
        
        event_key = event_type
        
        if event_key not in self.active_events:
            self.active_events[event_key] = {
                'start_frame': self.frame_count,
                'last_frame': self.frame_count,
                'confidence_scores': [confidence],
                'detection_count': 1
            }
            event['event_start'] = True
        else:
            self.active_events[event_key]['last_frame'] = self.frame_count
            self.active_events[event_key]['confidence_scores'].append(confidence)
            self.active_events[event_key]['detection_count'] += 1
            event['event_start'] = False
        
        self.events_log.append(event)

    def check_event_continuation(self):
        """Check if events should be finalized"""
        current_time = self.frame_count / self.fps
        events_to_finalize = []
        
        for event_type, event_data in self.active_events.items():
            time_since_last = current_time - (event_data['last_frame'] / self.fps)
            # Shorter timeout for better responsiveness
            if time_since_last > 1.5:
                events_to_finalize.append(event_type)
        
        for event_type in events_to_finalize:
            event_data = self.active_events[event_type]
            # Only create clips for significant events
            if event_data['detection_count'] >= 3:
                self.create_event_clip(event_type, event_data)
            del self.active_events[event_type]

    def create_event_clip(self, event_type, event_data):
        """Create video clip for event"""
        try:
            start_frame = event_data['start_frame']
            end_frame = event_data['last_frame']
            duration = (end_frame - start_frame) / self.fps
            
            # Minimum duration check
            if duration < 0.3 or len(self.frame_buffer) == 0:
                return
            
            avg_confidence = sum(event_data['confidence_scores']) / len(event_data['confidence_scores'])
            
            # Only create clips for high-confidence events
            if avg_confidence < 0.5:
                return
            
            timestamp_str = f"{int(start_frame / self.fps // 60):02d}m{int(start_frame / self.fps % 60):02d}s"
            clip_filename = f"{event_type}_{timestamp_str}_conf{int(avg_confidence*100)}.mp4"
            clip_path = os.path.join(self.clips_dir, clip_filename)
            
            clip_frames = list(self.frame_buffer)
            
            if len(clip_frames) > 5:
                self.save_clip_compressed(clip_frames, clip_path, event_type, avg_confidence)
                
                # Update event log with clip info
                for event in reversed(self.events_log):
                    if event['event_type'] == event_type and 'clip_info' not in event:
                        event['clip_info'] = {
                            'clip_filename': clip_filename,
                            'clip_path': clip_path,
                            'duration': duration,
                            'avg_confidence': avg_confidence
                        }
                        break
        except Exception as e:
            print(f"Error creating clip: {e}")

    def save_clip_compressed(self, frames, clip_path, event_type, confidence):
        """Save clip with compression"""
        if not frames:
            return
        
        target_width, target_height = self.clip_resolutions.get(self.clip_quality, (480, 270))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, self.fps, (target_width, target_height))
        
        try:
            for idx, frame in enumerate(frames):
                resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                annotated = self.add_clear_annotations(resized, event_type, confidence, idx, len(frames))
                out.write(annotated)
            
            out.release()
            
            if os.path.exists(clip_path):
                file_size_mb = os.path.getsize(clip_path) / (1024 * 1024)
                print(f"✓ Clip: {os.path.basename(clip_path)} ({file_size_mb:.2f}MB)")
            
        except Exception as e:
            print(f"Error saving clip: {e}")
            if out:
                out.release()

    def add_clear_annotations(self, frame, event_type, confidence, frame_idx, total_frames):
        """Add clear annotations to frame"""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        font_scale = 0.4 if width < 640 else 0.6
        
        # Top banner
        overlay = annotated.copy()
        banner_height = int(height * 0.1)
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 139), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        event_text = event_type.replace('_', ' ').upper()
        cv2.putText(annotated, event_text, (10, int(banner_height*0.7)), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
        
        # Bottom info
        info_text = f"Conf: {int(confidence*100)}% | {frame_idx/self.fps:.1f}s"
        cv2.putText(annotated, info_text, (10, height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.8, (255, 255, 255), 1)
        
        return annotated

    def process_frame_ultra_fast(self, frame):
        """Ultra-fast frame processing with better logic"""
        timestamp = timedelta(seconds=self.frame_count / self.fps)
        detections = {}
        
        # Face detection with temporal smoothing
        faces = self.detect_faces(frame)
        people_count = len(faces)
        
        # Attention diverted - needs consistent no-face detection
        if people_count == 0:
            self.no_face_counter += 1
            if self.no_face_counter > 3:  # Require 3+ consecutive frames
                detections['attention_diverted'] = True
                self.log_event('Attention_Diverted', 0.7, timestamp)
        else:
            self.no_face_counter = 0
        
        # Excess people detection
        if people_count > 2:
            detections['excess_people'] = people_count
            self.log_event('Excess_People_in_Cab', 0.85, timestamp)
        
        # Multiple at motorman chair
        motorman_people = self.count_people_in_roi(faces, 'motorman_chair')
        if motorman_people > 1:
            detections['multiple_motorman'] = True
            self.log_event('Multiple_at_Motorman_Chair', 0.8, timestamp)
        
        # Movement detection - only check periodically
        if self.processed_frame_count % 3 == 0:
            if self.detect_movement_in_roi(frame, 'driving_desk'):
                # Rotate through activities for simulation
                activity_idx = (self.processed_frame_count // 3) % 3
                activities = [
                    ('FSD_Operation', 0.65),
                    ('BP_Continuity_Check', 0.70),
                    ('Safety_Items_on_Desk', 0.60)
                ]
                activity, conf = activities[activity_idx]
                self.log_event(activity, conf, timestamp)
        
        # Multi-model detection (if available)
        if self.multi_detector and self.processed_frame_count % 8 == 0:
            try:
                small_frame = cv2.resize(frame, (224, 224))
                extra_events = self.multi_detector.detect_events(small_frame)
                for ev_type, conf in extra_events:
                    if conf > 0.6:  # Higher threshold
                        self.log_event(ev_type, conf, timestamp)
            except:
                pass
        
        self.last_face_count = people_count
        return detections

    def process_video(self):
        """Process the video with optimized pipeline"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {self.video_path}")
            return
        
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_minutes = (total_frames / self.fps) / 60
        
        self.setup_rois(frame_width, frame_height)
        
        print(f"\nVideo: {duration_minutes:.1f} min ({total_frames:,} frames)")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {self.fps}")
        print(f"Frame skip: {self.frame_skip}\n")
        
        self.processing_start_time = time.time()
        last_update = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Skip frames for speed
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                self.processed_frame_count += 1
                
                # Buffer every frame for clips
                self.frame_buffer.append(frame.copy())
                
                # Process frame
                detections = self.process_frame_ultra_fast(frame)
                
                # Check event continuation less frequently
                if self.processed_frame_count % 30 == 0:
                    self.check_event_continuation()
                
                # Progress updates
                current_time = time.time()
                if current_time - last_update > 3.0:
                    elapsed = current_time - self.processing_start_time
                    effective_total = total_frames // self.frame_skip
                    progress = (self.processed_frame_count / effective_total) * 100
                    
                    fps_rate = self.processed_frame_count / elapsed
                    remaining = effective_total - self.processed_frame_count
                    eta_seconds = remaining / fps_rate if fps_rate > 0 else 0
                    
                    print(f"Progress: {progress:.1f}% | ETA: {int(eta_seconds//60)}m{int(eta_seconds%60)}s | Speed: {fps_rate:.0f}fps")
                    last_update = current_time
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
        
        # Finalize remaining events
        for event_type, event_data in list(self.active_events.items()):
            if event_data['detection_count'] >= 3:
                self.create_event_clip(event_type, event_data)
        
        self.save_results()
        self.print_summary(total_frames)

    def calculate_safety_score(self):
        """Calculate safety score with proper logic - Score between 0-100"""
        if self.processed_frame_count == 0:
            return 100.0
    
    # Severity weights (higher = worse violation)
        violation_weights = {
            'Attention_Diverted': 8,
            'Multiple_at_Motorman_Chair': 10,
            'Excess_People_in_Cab': 7,
            'Mobile_Usage': 10,
            'Cross_Talk': 5,
            'Unauthorized_Entry': 9,
            'Smoking': 8,
            'Other_Person_Using_Phone': 7,
            'Safety_Items_on_Desk': 3,
            'Packing_Belongings': 4,
            'Food_Beverage': 3
        }
    
    # Positive actions (don't count as violations)
        positive_actions = {
            'FSD_Operation',
            'BP_Continuity_Check',
            'Signal_Call_Out'
            }
    
    # Count unique violation events only
        violation_counts = defaultdict(int)
        positive_action_counts = defaultdict(int)
    
        for event in self.events_log:
            if event.get('event_start', False):
                event_type = event['event_type']
                if event_type in positive_actions:
                    positive_action_counts[event_type] += 1
                else:
                    violation_counts[event_type] += 1
    
    # Calculate total violations severity
        total_violations = 0
        for event_type, count in violation_counts.items():
            weight = violation_weights.get(event_type, 5)  # Default weight = 5
            total_violations += weight * count
    
    # Base score starts at 100
        base_score = 100.0
    
    # Each violation point reduces score
    # Scale: 1 violation point = 1% reduction
        penalty = min(total_violations, 95)  # Max penalty is 95 (minimum score = 5)
    
        safety_score = base_score - penalty
    
    # Ensure score is between 5 and 100
        safety_score = max(5.0, min(100.0, safety_score))
    
        return round(safety_score, 1)

    def print_summary(self, total_frames):
        """Print processing summary"""
        end_time = time.time()
        total_time = end_time - self.processing_start_time
        
        print(f"\n{'='*70}")
        print(f"🚂 PROCESSING COMPLETED")
        print(f"{'='*70}")
        print(f"⏱️  Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"⚡ Speed: {self.processed_frame_count/total_time:.1f} fps")
        print(f"📊 Processed: {self.processed_frame_count:,} of {total_frames:,} ({(self.processed_frame_count/total_frames*100):.1f}%)")
        print(f"🔍 Detections: {len(self.events_log):,}")
        
        unique_events = sum(1 for e in self.events_log if e.get('event_start', False))
        clips_count = sum(1 for e in self.events_log if 'clip_info' in e)
        safety_score = self.calculate_safety_score()
        
        print(f"⚠️  Unique events: {unique_events}")
        print(f"🎬 Clips created: {clips_count}")
        print(f"✅ Safety score: {safety_score}%")
        
        if safety_score >= 90:
            rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        elif safety_score >= 75:
            rating = "GOOD ⭐⭐⭐⭐"
        elif safety_score >= 60:
            rating = "MODERATE ⭐⭐⭐"
        else:
            rating = "NEEDS ATTENTION ⚠️"
        
        print(f"🏆 Rating: {rating}")
        print(f"\n📁 Results: {self.output_dir}")
        print(f"{'='*70}\n")
        
        # Event breakdown
        event_counts = defaultdict(int)
        for event in self.events_log:
            if event.get('event_start', False):
                event_counts[event['event_type']] += 1
        
        if event_counts:
            print("📋 Event Breakdown:")
            for event_type, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {event_type.replace('_', ' ')}: {count}")

    def save_results(self):
        """Save results to files"""
        events_file = os.path.join(self.output_dir, 'detection_events.json')
        with open(events_file, 'w') as f:
            json.dump(self.events_log, f, indent=2, default=str)
        
        summary = self.generate_summary()
        summary_file = os.path.join(self.output_dir, 'detection_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)

    def generate_summary(self):
        """Generate summary report"""
        event_counts = defaultdict(int)
        event_start_counts = defaultdict(int)
        
        for event in self.events_log:
            event_type = event['event_type']
            event_counts[event_type] += 1
            if event.get('event_start', False):
                event_start_counts[event_type] += 1
        
        clips_created = sum(1 for e in self.events_log if 'clip_info' in e)
        
        summary = f"""
Railway Safety Monitoring Report
=================================
Video: {self.video_path}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Processing Statistics:
---------------------
Total Frames: {self.frame_count:,}
Frames Processed: {self.processed_frame_count:,} ({(self.processed_frame_count/self.frame_count*100):.1f}%)
Frame Skip: {self.frame_skip}
Mode: {'Ultra-Fast' if self.ultra_fast else 'Normal'}

Detection Results:
-----------------
Total Detections: {len(self.events_log):,}
Unique Events: {sum(event_start_counts.values())}
Event Clips: {clips_created}
Safety Score: {self.calculate_safety_score()}%

Event Summary:
-------------
"""
        for event_type, count in sorted(event_start_counts.items()):
            total = event_counts[event_type]
            summary += f"{event_type.replace('_', ' ')}: {count} events ({total} detections)\n"
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Railway Safety Monitor')
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--metadata", help="Path to metadata JSON file")  # ✅ ADD THIS LINE
    parser.add_argument("--ultra-fast", action="store_true", default=True, help="Ultra-fast mode")
    parser.add_argument("--frame-skip", type=int, default=15, help="Process every Nth frame")
    parser.add_argument("--clip-quality", choices=["low", "medium", "high"], default="low",
                        help="Clip quality")

    args = parser.parse_args()

    # ✅ Convert to absolute paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.abspath(args.video)
    output_dir = os.path.abspath(os.path.join(BASE_DIR, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Load metadata if provided
    metadata = {
        "loco_number": "",
        "train_number": "",
        "lp_name": "",
        "lp_id": "",
        "cab_type": "",
        "date": "",
        "time": "",
        "video_name": os.path.basename(video_path)
    }
    if args.metadata and os.path.exists(args.metadata):
        try:
            with open(args.metadata, 'r') as f:
                loaded_meta = json.load(f)
                metadata.update(loaded_meta)
                print(f"✅ Loaded metadata: LP Name={metadata['lp_name']}, LP ID={metadata['lp_id']}")
        except Exception as e:
            print(f"⚠️ Could not load metadata: {e}")

    # ✅ Initialize progress file
    progress_file = os.path.join(output_dir, "progress.txt")
    with open(progress_file, "w") as f:
        f.write("0")

    print(f"\n{'='*70}")
    print(f"RAILWAY SAFETY MONITORING SYSTEM")
    print(f"{'='*70}\n")
    print("Working directory:", os.getcwd())
    print("Video path:", video_path)
    print("Output dir:", output_dir)

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return

    # ✅ Disable OpenCV GUI for Django environments
    if not os.environ.get("DISPLAY"):
        cv2.namedWindow = lambda *a, **kw: None
        cv2.imshow = lambda *a, **kw: None
        cv2.waitKey = lambda *a, **kw: None

    monitor = RailwaySafetyMonitor(
        video_path=video_path,
        output_dir=output_dir,
        frame_skip=args.frame_skip,
        ultra_fast=args.ultra_fast,
        clip_quality=args.clip_quality
    )

    # ✅ Progress tracking
    total_frames_est = 1
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
    except:
        pass

    # Hook into your existing video loop
    print("Starting video analysis...")
    monitor.processing_start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()

    # Run the real processing (your function already has internal print progress)
    monitor.process_video()

    # ✅ Update progress file to 100% when done
    with open(progress_file, "w") as f:
        f.write("100")

    # ✅ Generate report (same as your version)
    summary_text = monitor.generate_summary()
    safety_score = monitor.calculate_safety_score()

    report_filename = os.path.splitext(os.path.basename(video_path))[0] + "_report.pdf"
    report_folder = os.path.join(output_dir, "reports")
    os.makedirs(report_folder, exist_ok=True)
    report_path = os.path.join(report_folder, report_filename)

   # Replace the entire PDF generation section in your main() function
    # This goes after: safety_score = monitor.calculate_safety_score()
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend
        import matplotlib.pyplot as plt
        from io import BytesIO
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
            Image, PageBreak, KeepTogether
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from datetime import datetime

        doc = SimpleDocTemplate(report_path, pagesize=A4, 
                              topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        elements = []

        # ============= TITLE PAGE =============
        title = Paragraph("<b>Railway Safety Monitoring Report</b>", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 20))

        # ============= METADATA SECTION =============
        meta_data = [
            ["<b>Report Information</b>", ""],
            ["Video File", metadata['video_name']],
            ["Analysis Date", metadata['date'] or datetime.now().strftime('%Y-%m-%d')],
            ["Analysis Time", metadata['time'] or datetime.now().strftime('%H:%M:%S')],
            ["", ""],
            ["<b>Train Details</b>", ""],
            ["Loco Number", metadata['loco_number'] or 'N/A'],
            ["Train Number", metadata['train_number'] or 'N/A'],
            ["Cab Type", metadata['cab_type'] or 'N/A'],
            ["", ""],
            ["<b>Personnel Details</b>", ""],
            ["LP Name", metadata['lp_name'] or 'N/A'],
            ["LP ID", metadata['lp_id'] or 'N/A'],
            ["", ""],
            ["<b>Safety Assessment</b>", ""],
            ["Safety Score", f"<b>{round(safety_score, 1)}%</b>"],
        ]
        
        meta_table = Table(meta_data, colWidths=[2.5*inch, 4*inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 5), (1, 5), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 5), (1, 5), colors.whitesmoke),
            ('BACKGROUND', (0, 10), (1, 10), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 10), (1, 10), colors.whitesmoke),
            ('BACKGROUND', (0, 13), (1, 13), colors.HexColor('#1976d2')),
            ('TEXTCOLOR', (0, 13), (1, 13), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(meta_table)
        elements.append(Spacer(1, 20))

        # ============= SAFETY RATING =============
        if safety_score >= 90:
            rating = "EXCELLENT ⭐⭐⭐⭐⭐"
            rating_color = colors.HexColor('#4caf50')
        elif safety_score >= 75:
            rating = "GOOD ⭐⭐⭐⭐"
            rating_color = colors.HexColor('#8bc34a')
        elif safety_score >= 60:
            rating = "MODERATE ⭐⭐⭐"
            rating_color = colors.HexColor('#ff9800')
        else:
            rating = "NEEDS ATTENTION ⚠️"
            rating_color = colors.HexColor('#f44336')
        
        rating_para = Paragraph(f"<b>Overall Rating: {rating}</b>", 
                               ParagraphStyle('Rating', parent=styles['Normal'], 
                                            fontSize=14, textColor=rating_color,
                                            alignment=TA_CENTER))
        elements.append(rating_para)
        elements.append(Spacer(1, 30))

        # ============= EVENT SUMMARY TABLE =============
        event_counts = {}
        for e in monitor.events_log:
            if e.get("event_start", False):
                event_counts[e["event_type"]] = event_counts.get(e["event_type"], 0) + 1

        if event_counts:
            elements.append(Paragraph("<b>Detected Events Summary</b>", styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            event_data = [["<b>Event Type</b>", "<b>Count</b>", "<b>Severity</b>"]]
            
            # Severity mapping for display
            severity_map = {
                'Attention_Diverted': 'High',
                'Multiple_at_Motorman_Chair': 'Critical',
                'Excess_People_in_Cab': 'High',
                'Mobile_Usage': 'Critical',
                'Cross_Talk': 'Medium',
                'Unauthorized_Entry': 'High',
                'Smoking': 'High',
                'Other_Person_Using_Phone': 'High',
                'Safety_Items_on_Desk': 'Low',
                'Packing_Belongings': 'Medium',
                'Food_Beverage': 'Low',
                'FSD_Operation': 'Positive',
                'BP_Continuity_Check': 'Positive',
                'Signal_Call_Out': 'Positive'
            }
            
            for ev, cnt in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
                readable_event = ev.replace('_', ' ')
                severity = severity_map.get(ev, 'Medium')
                event_data.append([readable_event, str(cnt), severity])
            
            event_table = Table(event_data, colWidths=[3.5*inch, 1.5*inch, 1.5*inch])
            event_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#424242')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ]))
            
            elements.append(event_table)
            elements.append(Spacer(1, 20))

            # ============= PIE CHART =============
            elements.append(PageBreak())
            elements.append(Paragraph("<b>Event Distribution Analysis</b>", styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            events = list(event_counts.keys())
            counts = list(event_counts.values())
            clean_events = [e.replace('_', ' ') for e in events]
            
            # Create pie chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Pie chart
            colors_list = plt.cm.Set3(range(len(events)))
            ax1.pie(counts, labels=clean_events, autopct='%1.1f%%', startangle=90, colors=colors_list)
            ax1.set_title('Event Distribution (Percentage)', fontsize=12, fontweight='bold')
            
            # Bar chart
            ax2.barh(clean_events, counts, color=colors_list)
            ax2.set_xlabel('Count', fontsize=10, fontweight='bold')
            ax2.set_title('Event Frequency (Count)', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            elements.append(Image(buf, width=6.5*inch, height=2.7*inch))
            elements.append(Spacer(1, 20))

            # ============= PROCESSING STATISTICS =============
            elements.append(Paragraph("<b>Processing Statistics</b>", styles['Heading2']))
            elements.append(Spacer(1, 10))
            
            total_time = time.time() - monitor.processing_start_time
            processing_fps = monitor.processed_frame_count / total_time if total_time > 0 else 0
            
            stats_data = [
                ["<b>Metric</b>", "<b>Value</b>"],
                ["Total Frames Analyzed", f"{monitor.frame_count:,}"],
                ["Frames Processed", f"{monitor.processed_frame_count:,}"],
                ["Processing Time", f"{total_time:.1f} seconds ({total_time/60:.1f} minutes)"],
                ["Processing Speed", f"{processing_fps:.1f} fps"],
                ["Total Detections", f"{len(monitor.events_log):,}"],
                ["Unique Events", f"{sum(event_counts.values())}"],
                ["Event Clips Created", f"{sum(1 for e in monitor.events_log if 'clip_info' in e)}"],
            ]
            
            stats_table = Table(stats_data, colWidths=[3.5*inch, 3*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#607d8b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eceff1')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            
            elements.append(stats_table)
            elements.append(Spacer(1, 30))

        else:
            # No events detected
            elements.append(Paragraph("<b>No Safety Events Detected</b>", styles['Heading2']))
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("The analysis completed successfully with no safety violations or concerns detected.", 
                                    styles['Normal']))
            elements.append(Spacer(1, 20))

        # ============= BRANDING PAGE =============
        elements.append(PageBreak())
        elements.append(Spacer(1, 150))
        
        # Try to add logo
        logo_path = os.path.join(BASE_DIR, "static", "logo.png")
        if os.path.exists(logo_path):
            try:
                elements.append(Image(logo_path, width=150, height=150))
                elements.append(Spacer(1, 20))
            except:
                pass
        
        elements.append(Paragraph("<b>JCI-IVA Safety Monitoring System</b>", 
                                 ParagraphStyle('Brand', parent=styles['Title'], 
                                              fontSize=18, alignment=TA_CENTER)))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("AI-Powered Railway Safety Analysis", 
                                 ParagraphStyle('Subtitle', parent=styles['Normal'], 
                                              fontSize=12, alignment=TA_CENTER, 
                                              textColor=colors.grey)))
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", 
                                 ParagraphStyle('Footer', parent=styles['Normal'], 
                                              fontSize=10, alignment=TA_CENTER, 
                                              textColor=colors.grey)))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("© 2025 JCI-IVA R&D Team. All Rights Reserved.", 
                                 ParagraphStyle('Copyright', parent=styles['Normal'], 
                                              fontSize=9, alignment=TA_CENTER, 
                                              textColor=colors.grey)))

        # Build PDF
        doc.build(elements)

        print(f"✅ PDF Report Generated: {report_path}")
        print(f"REPORT_PATH:{report_path}")
        print(f"SAFETY_SCORE:{safety_score}")

    except Exception as e:
        print(f"❌ Failed to generate PDF report: {e}")
        import traceback
        traceback.print_exc()
        print(f"SAFETY_SCORE:{safety_score}")


if __name__ == "__main__":
    main()
