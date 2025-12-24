from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from myapp import views   # import views directly from your app
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/upload-video/', views.upload_video, name="upload_video"),
    path('api/start-analysis/', views.start_analysis, name="start_analysis"),
    path("api/get-progress/", views.get_progress, name="get_progress"),
    path("api/get-summary/", views.get_summary, name="get_summary"),
    path('', views.gui, name="gui"),
    path('api/save-lp-profile/', views.save_lp_profile, name='save_lp_profile'),
    path('api/login/', views.login, name='login'),
    ] 
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
from django.contrib.auth import views as auth_views
