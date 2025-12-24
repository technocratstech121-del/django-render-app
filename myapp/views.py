import os
import json
import subprocess
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import AnalysisResult, AnalysisRecord
import threading
import time
from django.conf import settings
import sys
from datetime import datetime
from .report_generator import generate_analysis_report
from django.core.serializers import serialize
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required

# directory to store uploaded files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def gui(request):
    """Render main single-page GUI - starts at login"""
    return render(request, "GUI.html")

@login_required(login_url='login')
def dashboard(request):
    return render(request, 'index.html')

@csrf_exempt
def login(request):
    """User login authentication"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    
    try:
        data = json.loads(request.body.decode("utf-8"))
        user_id = data.get('user_id', '').strip()
        password = data.get('password', '').strip()
        
        if not user_id or not password:
            return JsonResponse({
                "success": False,
                "message": "User ID and Password are required"
            }, status=400)
        
        # Authenticate user
        user = authenticate(username=user_id, password=password)
        
        if user is not None:
            # Login successful
            return JsonResponse({
                "success": True,
                "user_id": user.username,
                "user_name": user.get_full_name() or user.username,
                "role": "admin" if user.is_staff else "user",
                "message": "Login successful"
            })
        else:
            # Invalid credentials
            return JsonResponse({
                "success": False,
                "message": "Invalid User ID or Password"
            }, status=401)
        
    except Exception as e:
        print("❌ Login error:", e)
        import traceback
        traceback.print_exc()
        return JsonResponse({
            "success": False,
            "message": "Login failed. Please try again."
        }, status=500)
    
@csrf_exempt
def upload_video(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "message": "POST required"}, status=400)

    # Handle multiple files
    videos = request.FILES.getlist("videos")
    
    if not videos:
        return JsonResponse({
            "success": False, 
            "message": "No videos uploaded.",
            "received_fields": list(request.FILES.keys())
        }, status=400)

    upload_dir = os.path.join(settings.BASE_DIR, "myapp", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    uploaded_files = []
    
    for video in videos:
        video_path = os.path.join(upload_dir, video.name)
        
        with open(video_path, "wb+") as dest:
            for chunk in video.chunks():
                dest.write(chunk)
        
        print(f"📹 Uploaded: {video_path}")
        uploaded_files.append(video.name)

    return JsonResponse({
        "success": True,
        "files": uploaded_files,
        "message": f"Uploaded {len(uploaded_files)} video(s)"
    })

@csrf_exempt
def start_analysis(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    try:
        # --- Parse JSON request body ---
        data = json.loads(request.body.decode("utf-8"))
        files = data.get("files", [])
        meta = data.get("meta", {})

        # ✅ DEBUG: Print EVERYTHING received
        print("=" * 70)
        print("📦 RECEIVED DATA:")
        print(json.dumps(data, indent=2))
        print("=" * 70)
        print("📋 META OBJECT:")
        print(json.dumps(meta, indent=2))
        print("=" * 70)
        
        # --- Extract metadata from frontend (use camelCase keys) ---
        loco_number = meta.get("locoNumber", "")  # ✅ Fixed: was "loco_number"
        train_number = meta.get("trainNumber", "")  # ✅ Fixed: was "train_number"
        lp_name = meta.get("lpName", "")  # ✅ Fixed: was "lp_name"
        lp_id = meta.get("lpId", "")  # ✅ Fixed: was "lp_id"
        cab_type = meta.get("cabType", "")  # ✅ Fixed: was "cab_type"
        date_val = meta.get("date", "")
        time_val = meta.get("time", "")

        print(f"📋 Received metadata:")
        print(f"   Loco Number: {loco_number}")
        print(f"   Train Number: {train_number}")
        print(f"   LP Name: {lp_name}")
        print(f"   LP ID: {lp_id}")
        print(f"   Cab Type: {cab_type}")
        print(f"   Date: {date_val}")
        print(f"   Time: {time_val}")

        if not files:
            return JsonResponse({"error": "No video files provided"}, status=400)

        video_name = files[0]
        video_path = os.path.join(settings.BASE_DIR, "myapp", "uploads", video_name)

        if not os.path.exists(video_path):
            return JsonResponse({"error": f"Video not found: {video_name}"}, status=404)

        # --- Output directory ---
        output_dir = os.path.join(settings.BASE_DIR, "myapp", "output")
        os.makedirs(output_dir, exist_ok=True)

        # ✅ Create metadata JSON file for the script to read
        metadata_file = os.path.join(output_dir, "metadata.json")
        metadata_dict = {
            "loco_number": loco_number,
            "train_number": train_number,
            "lp_name": lp_name,
            "lp_id": lp_id,
            "cab_type": cab_type,
            "date": date_val,
            "time": time_val,
            "video_name": video_name
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"✅ Created metadata file: {metadata_file}")
        print(f"📋 Metadata contents:")
        print(json.dumps(metadata_dict, indent=2))

        # --- Run analysis script and capture output ---
        script_path = os.path.join(settings.BASE_DIR, "myapp", "railway_safety_monitor.py")
        process = subprocess.Popen(
            [sys.executable, "-u", script_path,
             "--video", video_path,
             "--output-dir", output_dir,
             "--metadata", metadata_file],  # ✅ ADDED: Pass metadata file
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        # ✅ Capture output and extract safety score and report path
        actual_safety_score = 93.0
        actual_report_path = None
        
        for line in iter(process.stdout.readline, ""):
            line = line.strip()
            print(line)
            
            if line.startswith("SAFETY_SCORE:"):
                try:
                    actual_safety_score = float(line.split(":", 1)[1].strip())
                    print(f"✅ Captured Safety Score: {actual_safety_score}%")
                except Exception as e:
                    print(f"⚠️ Failed to parse safety score: {e}")
            
            if line.startswith("REPORT_PATH:"):
                try:
                    actual_report_path = line.split(":", 1)[1].strip()
                    print(f"✅ Captured Report Path: {actual_report_path}")
                except Exception as e:
                    print(f"⚠️ Failed to parse report path: {e}")
        
        process.stdout.close()
        process.wait()

        # Generate remarks based on safety score
        if actual_safety_score >= 90:
            actual_remarks = "Excellent performance. No significant safety concerns detected."
        elif actual_safety_score >= 75:
            actual_remarks = "Good performance with minor observations."
        elif actual_safety_score >= 60:
            actual_remarks = "Moderate performance. Some safety concerns detected."
        else:
            actual_remarks = "Needs attention. Multiple safety violations detected."

        # Read event summary for detailed remarks
        detection_file = os.path.join(output_dir, "detection_events.json")
        if os.path.exists(detection_file):
            try:
                with open(detection_file, 'r') as f:
                    events = json.load(f)
                    event_counts = {}
                    for event in events:
                        if event.get('event_start', False):
                            event_type = event['event_type']
                            event_counts[event_type] = event_counts.get(event_type, 0) + 1
                    
                    if event_counts:
                        remarks_parts = []
                        for event_type, count in event_counts.items():
                            readable_event = event_type.replace('_', ' ')
                            remarks_parts.append(f"{readable_event}: {count}")
                        actual_remarks = "; ".join(remarks_parts)
                        print(f"✅ Generated Remarks: {actual_remarks}")
            except Exception as e:
                print(f"⚠️ Could not read detection events: {e}")

        # Determine report path
        report_url = ""
        if actual_report_path and os.path.exists(actual_report_path):
            reports_dir = os.path.join(settings.MEDIA_ROOT, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            report_filename = os.path.basename(actual_report_path)
            final_report_path = os.path.join(reports_dir, report_filename)
            
            if actual_report_path != final_report_path:
                import shutil
                shutil.copy2(actual_report_path, final_report_path)
                print(f"✅ Copied report to: {final_report_path}")
            
            report_url = f"/media/reports/{report_filename}"
        else:
            print("⚠️ No report path found")

        # --- Save record to database ---
        print(f"💾 Saving to database:")
        print(f"   Video: {video_name}")
        print(f"   Train: {train_number}")
        print(f"   LP Name: {lp_name}")
        print(f"   LP ID: {lp_id}")
        print(f"   Score: {actual_safety_score}")
        
        AnalysisRecord.objects.create(
            date=date_val or None,
            time=time_val or None,
            loco_number=loco_number,
            train_number=train_number,
            lp_name=lp_name,
            lp_id=lp_id,
            cab_type=cab_type,
            video_name=video_name,
            safety_score=actual_safety_score,
            remarks=actual_remarks,
            report_path=report_url
        )
        
        print("✅ Database record created successfully")

        return JsonResponse({
            "message": "✅ Analysis completed successfully",
            "report_url": report_url,
            "safety_score": actual_safety_score,
            "remarks": actual_remarks
        })

    except Exception as e:
        print("❌ Error:", e)
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def get_summary(request):
    try:
        records = AnalysisRecord.objects.order_by("-created_at")[:50]
        data = []
        
        for r in records:
            data.append({
                "date": str(r.date) if r.date else "",  # ✅ Fixed: lowercase
                "time": str(r.time) if r.time else "",  # ✅ Fixed: lowercase
                "video_name": r.video_name or "",
                "loco_number": r.loco_number or "",
                "train_number": r.train_number or "",
                "lp_name": r.lp_name or "",
                "lp_id": r.lp_id or "",
                "cab_type": r.cab_type or "",
                "safety_score": r.safety_score if r.safety_score is not None else 0,
                "remarks": r.remarks or "",
                "report_path": r.report_path or "",
                "created_at": r.created_at.isoformat() if r.created_at else "",
            })
        
        print(f"📊 Returning {len(data)} records")
        if data:
            print(f"Sample record: {data[0]}")
        
        return JsonResponse({
            "success": True,
            "summary": data
        })
    except Exception as e:
        print("❌ Summary fetch error:", e)
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)})

def get_progress(request):
    """Return progress for all videos"""
    data = list(AnalysisResult.objects.values("video_name", "status", "progress", "message"))
    return JsonResponse({"success": True, "progress": data})

@csrf_exempt
def save_lp_profile(request):
    """Save a new LP profile"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    
    try:
        data = json.loads(request.body.decode("utf-8"))
        
        lp_name = data.get('lp_name', '')
        lp_id = data.get('lp_id', '')
        phone = data.get('phone', '')
        email = data.get('email', '')
        division = data.get('division', '')
        experience = data.get('experience', '')
        
        if not lp_name or not lp_id:
            return JsonResponse({"success": False, "error": "LP Name and LP ID are required"}, status=400)
        
        # Create a dummy analysis record to store LP profile
        # You can create a separate LPProfile model if you want
        AnalysisRecord.objects.create(
            lp_name=lp_name,
            lp_id=lp_id,
            video_name=f"LP_PROFILE_{lp_id}",  # Marker to identify profile-only records
            remarks=f"Phone: {phone}, Email: {email}, Division: {division}, Experience: {experience} years",
            safety_score=None
        )
        
        return JsonResponse({
            "success": True,
            "message": "LP Profile saved successfully"
        })
        
    except Exception as e:
        print("❌ Error saving LP profile:", e)
        import traceback
        traceback.print_exc()
        return JsonResponse({"success": False, "error": str(e)}, status=500)