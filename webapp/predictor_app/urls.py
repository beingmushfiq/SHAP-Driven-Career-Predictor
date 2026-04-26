from django.urls import path
from .views import IndexView, PredictView, GlobalAnalysisView, AboutView

# Note: ResultView is handled within PredictView.post or we can split it.
# For simplicity, let's keep the views as defined.

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('analysis/', GlobalAnalysisView.as_view(), name='analysis'),
    path('about/', AboutView.as_view(), name='about'),
]
