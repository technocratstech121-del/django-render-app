from django.db import models
from django.utils import timezone

class AnalysisRecord(models.Model):
    date = models.DateField(null=True, blank=True)  
    time = models.TimeField(null=True, blank=True)
    loco_number = models.CharField(max_length=50,default='', blank=True)
    train_number = models.CharField(max_length=50,default='', blank=True)
    lp_name = models.CharField(max_length=100,default='', blank=True)
    lp_id = models.CharField(max_length=50,default='', blank=True)
    cab_type = models.CharField(max_length=50,default='', blank=True)
    safety_score = models.FloatField(null=True, blank=True)
    remarks = models.TextField(default='', blank=True)
    report_path = models.CharField(max_length=300,default='', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    video_name = models.CharField(max_length=255)
    status = models.CharField(max_length=50, default="Pending")  # Pending / Processing / Done / Error
    result_summary = models.TextField(blank=True, null=True)
    report_file = models.FileField(upload_to='reports/', blank=True, null=True)

    def __str__(self):
        return f"{self.video_name} - {self.status}"

# Create your models here.
class AnalysisResult(models.Model):
    video_name = models.CharField(max_length=255)
    loco_number = models.CharField(max_length=100, blank=True, null=True)
    train_number = models.CharField(max_length=100, blank=True, null=True)
    lp_name = models.CharField(max_length=100, blank=True, null=True)
    lp_id = models.CharField(max_length=100, blank=True, null=True)
    date = models.DateField(blank=True, null=True)
    time = models.TimeField(blank=True, null=True)
    cab_type = models.CharField(max_length=50, default="unknown")
    status = models.CharField(max_length=50)
    safety_score = models.FloatField(default=0)
    report_path = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # ✅ add progress tracking
    progress = models.IntegerField(default=0)  # percentage (0–100)
    message = models.CharField(max_length=255, blank=True, null=True)
    def __str__(self):
        return f"{self.video_name} - {self.status}"
