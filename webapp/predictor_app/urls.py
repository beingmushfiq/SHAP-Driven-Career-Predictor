from django.urls import path
from .views import IndexView, PredictView, GlobalAnalysisView, AboutView, SystemStatusView, ComparisonDashboardView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('analysis/', GlobalAnalysisView.as_view(), name='analysis'),
    path('compare/', ComparisonDashboardView.as_view(), name='compare'),
    path('about/', AboutView.as_view(), name='about'),
    path('system-status/', SystemStatusView.as_view(), name='system_status'),
]
