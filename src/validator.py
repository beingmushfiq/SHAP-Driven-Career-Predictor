"""
Background to Career Validation Layer for Career Predictor.

Checks educational/field background and skill levels to determine if the predicted
career cluster is aligned, flagging anomalies or potential mismatch indicators.
"""

from typing import Dict, List, Tuple
from src.config import Config


class CareerValidator:
    """
    Validates predictions against user background and skills.
    """

    # Rules mapping: Field -> List of aligned Career Clusters
    FIELD_ALIGNMENTS = {
        'computer science': ['Software Engineer', 'Data & AI Specialist'],
        'engineering': ['Engineer', 'Architect & Planner'],
        'medicine': ['Doctor & Surgeon', 'Healthcare Specialist'],
        'business': ['Business Manager', 'Marketing Professional', 'Brand & Advertising'],
        'finance': ['Finance Professional', 'Investment & Insurance'],
        'marketing': ['Marketing Professional', 'Brand & Advertising', 'Business Manager'],
        'law': ['Legal Professional', 'Legal Support'],
        'education': ['Educator', 'Counselor & Therapist'],
        'psychology': ['Psychologist', 'Counselor & Therapist'],
        'biology': ['Biologist', 'Healthcare Specialist'],
        'chemistry': ['Chemist', 'Healthcare Specialist'],
        'physics': ['Physicist', 'Engineer'],
        'architecture': ['Architect & Planner', 'Visual Artist'],
        'art': ['Visual Artist', 'Architect & Planner'],
        'music': ['Musician & Audio', 'Educator'],
    }

    # Skill requirement rules: Career Cluster -> Min skill levels required
    SKILL_REQUIREMENTS = {
        'Software Engineer': {'coding_skills': 3.0, 'problem_solving_skills': 3.0},
        'Data & AI Specialist': {'coding_skills': 3.0, 'analytical_skills': 3.5},
        'Engineer': {'analytical_skills': 3.0, 'problem_solving_skills': 3.0},
        'Doctor & Surgeon': {'gpa': 3.5, 'analytical_skills': 3.0},
        'Finance Professional': {'analytical_skills': 3.0},
        'Legal Professional': {'communication_skills': 3.5, 'analytical_skills': 3.0},
        'Marketing Professional': {'communication_skills': 3.0, 'presentation_skills': 3.0},
        'Musician & Audio': {'extracurricular_activities': 4.0},
    }

    @classmethod
    def validate_prediction(cls, form_data: Dict, predicted_career: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate predicted career against user background (field) and skills.

        Args:
            form_data: Raw input dictionary (form field keys).
            predicted_career: The career cluster predicted by the model.

        Returns:
            Tuple of (is_aligned: bool, warnings: List[str], suggestions: List[str]).
        """
        warnings = []
        suggestions = []
        is_aligned = True

        # Extract features
        field = str(form_data.get('field', '')).strip().lower()
        gpa = float(form_data.get('gpa', 0.0))
        
        # 1. Educational Field Alignment Check
        if field in cls.FIELD_ALIGNMENTS:
            aligned_clusters = cls.FIELD_ALIGNMENTS[field]
            if predicted_career not in aligned_clusters:
                is_aligned = False
                warnings.append(
                    f"Predicted career '{predicted_career}' does not typically align "
                    f"with your major/field of '{field.title()}'."
                )
                suggestions.append(
                    f"Consider paths more closely aligned with {field.title()}, "
                    f"such as: {', '.join(aligned_clusters)}."
                )

        # 2. Skill Requirements Check
        if predicted_career in cls.SKILL_REQUIREMENTS:
            reqs = cls.SKILL_REQUIREMENTS[predicted_career]
            for skill_name, min_val in reqs.items():
                # Form data might have space/casing differences
                val = float(form_data.get(skill_name, form_data.get(skill_name.replace('_', ' ').title(), 0.0)))
                if val < min_val:
                    is_aligned = False
                    warnings.append(
                        f"Your score for '{skill_name.replace('_', ' ').title()}' ({val}) "
                        f"is lower than typical requirements for '{predicted_career}' (min {min_val})."
                    )
                    suggestions.append(
                        f"Enhance your '{skill_name.replace('_', ' ').title()}' via projects or certified courses."
                    )

        # 3. GPA Mismatch Check for highly academic/selective fields
        if predicted_career in ['Doctor & Surgeon', 'Legal Professional'] and gpa < 3.0:
            is_aligned = False
            warnings.append(
                f"Your GPA ({gpa}) is below the competitive threshold for "
                f"entering '{predicted_career}'."
            )
            suggestions.append(
                "Focus on lifting academic scores or seek internship credentials to offset GPA."
            )

        return is_aligned, warnings, suggestions
