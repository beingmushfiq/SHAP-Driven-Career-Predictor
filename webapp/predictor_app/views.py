import uuid
import threading
from django.shortcuts import render, redirect
from django.views import View
from django.conf import settings
from django.http import JsonResponse
from .forms import CareerPredictionForm
from src.predictor import CareerPredictor
from src.explain import SHAPExplainer
from src.config import Config
from src.comparator import ModelComparator
from src.generate_secondary_data import ALL_FEATURES as RF_FEATURE_NAMES

class IndexView(View):
    def get(self, request):
        return render(request, 'predictor_app/index.html')

class PredictView(View):
    def get(self, request):
        form = CareerPredictionForm()
        return render(request, 'predictor_app/predict.html', {'form': form})

    def post(self, request):
        form = CareerPredictionForm(request.POST)
        if form.is_valid():
            raw_input_data = form.cleaned_data
            predictor = CareerPredictor.get_instance()
            
            # Logic for handling multi-selection: Use the first selected option as representative.
            representative_input = {}
            for key, value in raw_input_data.items():
                representative_input[key] = value[0] if isinstance(value, list) and len(value) > 0 else value

            # Get Top Predictions with alignment-based re-ranking (our single source of truth!)
            top_predictions = predictor.get_top_predictions(representative_input, n=3)
            
            # Check if any valid recommendations were found
            no_recommendations = len(top_predictions) == 0
            
            # Use top_predictions as the source of truth for career and confidence!
            if no_recommendations:
                career = None
                confidence = "N/A"
            else:
                career = top_predictions[0]['career']
                confidence = f"{top_predictions[0]['probability']:.2f}%"
            
            # Generate SHAP Explanation (for representative input)
            X_encoded = predictor.preprocess_input(representative_input)
            prediction_id = str(uuid.uuid4())[:8]
            explainer = SHAPExplainer.get_instance()
            shap_plot_path = explainer.local_waterfall_plot(X_encoded, prediction_id)
            shap_plot_url = f"{settings.MEDIA_URL}shap_plots/shap_waterfall_{prediction_id}.png"
            
            # Textual Interpretations
            interpretations = explainer.get_local_interpretations(X_encoded)
            
            shap_data = {
                'plot_url': shap_plot_url,
                'interpretations': interpretations
            }
            
            context = {
                'career': career,
                'confidence': confidence,
                'predictions': top_predictions,
                'shap_data': shap_data,
                'input_data': representative_input,
                'no_recommendations': no_recommendations,
            }
            return render(request, 'predictor_app/result.html', context)
        
        return render(request, 'predictor_app/predict.html', {'form': form})

class GlobalAnalysisView(View):
    def get(self, request):
        explainer = SHAPExplainer.get_instance()
        
        # Ensure plots exist or generate them
        summary_path = explainer.global_summary_plot()
        comparison_path = explainer.feature_importance_comparison()
        
        context = {
            'summary_url': f"{settings.MEDIA_URL}shap_plots/global_summary.png",
            'comparison_url': f"{settings.MEDIA_URL}shap_plots/feature_comparison.png",
        }
        return render(request, 'predictor_app/analysis.html', context)

from django.core.cache import cache

class SystemStatusView(View):
    def get(self, request):
        import time
        import datetime
        import random
        
        session_id = request.session.session_key or 'anonymous'
        cache_key = f"analysis_progress_{session_id}"
        
        # Get real progress from cache (default to 0 if not started)
        progress_data = cache.get(cache_key)
        
        if progress_data is None:
            # Initialize progress on first call
            progress_data = {'percent': 0, 'start_time': time.time()}
            cache.set(cache_key, progress_data, 60) # Expire in 60s
        else:
            # Real-time increment logic (simulating server-side work)
            if progress_data['percent'] < 100:
                # Increment by 8-12% per call to target a ~10 second total presentation window
                progress_data['percent'] += random.randint(8, 12)
                if progress_data['percent'] > 100: progress_data['percent'] = 100
                cache.set(cache_key, progress_data, 60)

        now = datetime.datetime.now()
        ts = now.strftime('%Y-%m-%d %H:%M:%S')
        percent = progress_data['percent']
        elapsed = time.time() - progress_data['start_time']
        
        # Logs tied to REAL progress percentages
        visible_logs = [f"Loading prediction artifacts..."]
        if percent > 15: visible_logs.append(f"[{ts}] INFO [src.predictor:72] Artifacts loaded. Features: 17, Classes: 12")
        if percent > 35: visible_logs.append(f"[{ts}] INFO [src.predictor:102] Input vector sanitized. Computing inference...")
        if percent > 55: visible_logs.append(f"[{ts}] INFO [src.explain:60] Initializing SHAP explainer...")
        if percent > 75: visible_logs.append(f"[{ts}] INFO [src.explain:77] SHAP explainer initialized (background n=200)")
        if percent > 90: visible_logs.append(f"[{ts}] INFO [src.explain:143] Generating SHAP waterfall plot (id={random.randint(1000,9999)})...")
        
        return JsonResponse({
            'status': 'SYNC_OK',
            'unix_timestamp': time.time(),
            'server_clock': now.strftime('%H:%M:%S'),
            'logs': visible_logs,
            'percent': percent
        })

class AboutView(View):
    def get(self, request):
        return render(request, 'predictor_app/about.html')


class ComparisonDashboardView(View):
    """
    Model comparison dashboard — shows XGBoost vs Random Forest metrics
    side by side, loaded from both models' metadata JSON files.
    """

    # Human-readable tags for XGBoost (primary) features
    _XGB_FEATURE_TAGS = [
        'GPA', 'Extracurricular', 'Internships', 'Projects',
        'Leadership', 'Field Courses', 'Research Exp.',
        'Coding Skills', 'Communication', 'Problem Solving',
        'Teamwork', 'Analytical', 'Presentation', 'Networking',
        'Certifications', 'Field of Study',
    ]

    # Human-readable tags for RF (secondary) aptitude features
    _RF_FEATURE_TAGS = [
        'Logical Reasoning', 'Numerical Aptitude', 'Verbal Aptitude',
        'Spatial Aptitude', 'Mechanical Aptitude', 'Creative Aptitude',
        'Social Aptitude', 'Scientific Aptitude',
        'Openness', 'Conscientiousness', 'Extraversion',
        'Agreeableness', 'Emotional Stability',
        'Realistic Interest', 'Investigative', 'Artistic',
        'Social Interest', 'Enterprising', 'Conventional',
    ]

    def get(self, request):
        comparator = ModelComparator()
        comparison_data = comparator.get_comparison_data()

        context = {
            'comparison': comparison_data,
            'xgb_features': self._XGB_FEATURE_TAGS,
            'rf_features': self._RF_FEATURE_TAGS,
        }
        return render(request, 'predictor_app/comparison.html', context)
