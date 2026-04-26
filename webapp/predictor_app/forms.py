from django import forms
from src.config import Config

class CareerPredictionForm(forms.Form):
    # Numerical features
    logical_quotient = forms.IntegerField(
        label="Logical Quotient (1-10)",
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 7'})
    )
    hackathons = forms.IntegerField(
        label="Number of Hackathons (0-10)",
        min_value=0, max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 3'})
    )
    coding_skills = forms.IntegerField(
        label="Coding Skills (1-10)",
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 8'})
    )
    public_speaking = forms.IntegerField(
        label="Public Speaking (1-10)",
        min_value=1, max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g. 6'})
    )

    # Categorical features - MultipleChoiceField for multi-select support
    self_learning = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['self_learning']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    extra_courses = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['extra_courses']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    certifications = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['certifications']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    workshops = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['workshops']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    reading_writing_skills = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['reading_writing_skills']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    memory_capability = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['memory_capability']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    interested_subjects = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['interested_subjects']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    interested_career = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['interested_career']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    company_type = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['company_type']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    senior_elder_advise = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['senior_elder_advise']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    book_general_genre = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['book_general_genre']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    management_technical = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['management_technical']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
    hard_smart_worker = forms.MultipleChoiceField(
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['hard_smart_worker']],
        widget=forms.Select(attrs={'class': 'form-select select-ticked-multi', 'multiple': 'multiple'})
    )
