from django import forms
from src.config import Config

class CareerPredictionForm(forms.Form):
    # Academic & Field (Step 1)
    field = forms.ChoiceField(
        label="Primary Field of Study",
        choices=[(opt, opt) for opt in Config.CATEGORICAL_OPTIONS['field']],
        widget=forms.Select(attrs={'class': 'form-select select2-basic', 'required': 'required'})
    )
    gpa = forms.FloatField(
        label="CGPA (Out of 4.0)",
        min_value=2.0, max_value=4.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': 'e.g. 3.75', 'required': 'required'})
    )
    field_specific_courses = forms.IntegerField(
        label="Field Specific Courses (0-10)",
        min_value=0, max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required'})
    )

    # Experience & Activities (Step 2)
    internships = forms.IntegerField(
        label="Number of Internships (0-5)",
        min_value=0, max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required'})
    )
    projects = forms.IntegerField(
        label="Number of Projects (0-10)",
        min_value=0, max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required'})
    )
    extracurricular_activities = forms.IntegerField(
        label="Extracurricular Activities (0-10)",
        min_value=0, max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'required': 'required'})
    )
    leadership_positions = forms.ChoiceField(
        label="Leadership Experience",
        choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-select', 'required': 'required'})
    )
    research_experience = forms.ChoiceField(
        label="Research Experience",
        choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-select', 'required': 'required'})
    )
    industry_certifications = forms.ChoiceField(
        label="Industry Certifications",
        choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-select', 'required': 'required'})
    )

    # Core Skills (Step 3) - Rating 0-5
    coding_skills = forms.IntegerField(
        label="Coding Skills (0-5)",
        min_value=0, max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Rate 0-5', 'required': 'required'})
    )
    communication_skills = forms.IntegerField(
        label="Communication Skills (0-5)",
        min_value=0, max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Rate 0-5', 'required': 'required'})
    )
    problem_solving_skills = forms.IntegerField(
        label="Problem Solving Skills (0-5)",
        min_value=0, max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Rate 0-5', 'required': 'required'})
    )
    teamwork_skills = forms.IntegerField(
        label="Teamwork Skills (0-5)",
        min_value=0, max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Rate 0-5', 'required': 'required'})
    )

    # Professional Skills (Step 4) - Rating 0-5
    analytical_skills = forms.IntegerField(
        label="Analytical Skills (0-5)",
        min_value=0, max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Rate 0-5', 'required': 'required'})
    )
    presentation_skills = forms.IntegerField(
        label="Presentation Skills (0-5)",
        min_value=0, max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Rate 0-5', 'required': 'required'})
    )
    networking_skills = forms.IntegerField(
        label="Networking Skills (0-5)",
        min_value=0, max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Rate 0-5', 'required': 'required'})
    )
