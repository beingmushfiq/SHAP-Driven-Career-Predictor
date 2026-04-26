import uuid
from django.shortcuts import render, redirect
from django.views import View
from django.conf import settings
from .forms import CareerPredictionForm
from src.predictor import CareerPredictor
from src.explain import SHAPExplainer
from src.config import Config

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
            
            # Logic for handling multi-selection: 
            # We aggregate probabilities across all selected items for each multi-select field.
            # To avoid exponential combinations, we use a 'representative' approach:
            # We predict for each selection in each field while keeping others constant (first selected).
            
            all_probs = []
            representative_input = {}
            for key, value in raw_input_data.items():
                representative_input[key] = value[0] if isinstance(value, list) and len(value) > 0 else value

            # Base prediction for representative input
            _, base_probs = predictor.predict(representative_input)
            all_probs.append(base_probs)

            # Additional predictions for other selections
            for key, value in raw_input_data.items():
                if isinstance(value, list) and len(value) > 1:
                    for extra_val in value[1:]:
                        temp_input = representative_input.copy()
                        temp_input[key] = extra_val
                        _, p = predictor.predict(temp_input)
                        all_probs.append(p)
            
            # Average probabilities
            import numpy as np
            avg_probs = np.mean(all_probs, axis=0)
            
            # Get final career and confidence
            class_names = predictor.get_class_names()
            career_idx = avg_probs.argmax()
            career = class_names[career_idx]
            confidence = avg_probs[career_idx] * 100
            
            # Get Top Predictions (sorted by avg_probs)
            top_indices = avg_probs.argsort()[-3:][::-1]
            top_predictions = [(class_names[i], avg_probs[i]) for i in top_indices]
            
            # 3. Generate SHAP Explanation (for representative input)
            X_encoded = predictor.preprocess_input(representative_input)
            prediction_id = str(uuid.uuid4())[:8]
            explainer = SHAPExplainer.get_instance()
            shap_plot_path = explainer.local_waterfall_plot(X_encoded, prediction_id)
            shap_plot_url = f"{settings.MEDIA_URL}shap_plots/shap_waterfall_{prediction_id}.png"
            
            context = {
                'career': career,
                'confidence': f"{confidence:.2f}%",
                'top_predictions': top_predictions,
                'shap_plot_url': shap_plot_url,
                'input_data': representative_input
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

class AboutView(View):
    def get(self, request):
        return render(request, 'predictor_app/about.html')
