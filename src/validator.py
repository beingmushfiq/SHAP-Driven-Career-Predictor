"""
Career Validation Layer for Career Predictor.

Checks educational/field background, skills, and interests to determine if the predicted
career cluster is well-aligned with user profile. Returns alignment scores and recommendations.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from src.config import Config


class CareerValidator:
    """
    Validates predictions against user background, skills, and interests.
    Provides alignment scores and recommendations.
    """

    @classmethod
    def compute_education_alignment_score(cls, field: str, predicted_career: str) -> float:
        """
        Compute education-career alignment score (0-100).
        
        Args:
            field: User's academic field
            predicted_career: Predicted career cluster
            
        Returns:
            Alignment score (0-100)
        """
        if not field:
            return 50.0  # Neutral score for empty field
        
        # Normalize field lookup: try exact match first, then title case
        field_str = str(field).strip()
        aligned_careers = Config.FIELD_CAREER_ALIGNMENT.get(field_str, None)
        
        if aligned_careers is None:
            # Try title case (e.g., 'computer science' -> 'Computer Science')
            field_title = field_str.title()
            aligned_careers = Config.FIELD_CAREER_ALIGNMENT.get(field_title, [])
        
        if not aligned_careers:
            return 50.0  # Neutral score for unmapped fields
        
        if predicted_career in aligned_careers:
            # Exact alignment: 95-100
            return 95.0
        
        # Partial alignment: check if categories overlap
        # (e.g., 'Engineer' and 'Software Engineer' both tech-heavy)
        partial_match_weight = 60.0
        
        return min(partial_match_weight, 75.0)

    @classmethod
    def compute_skill_alignment_score(cls, form_data: Dict, predicted_career: str) -> Tuple[float, List[str]]:
        """
        Compute skill-career alignment score (0-100) and identify gaps.
        
        Args:
            form_data: User input data with skill levels
            predicted_career: Predicted career cluster
            
        Returns:
            Tuple of (alignment_score, gap_list)
        """
        # Get skill requirements from config
        reqs = Config.CAREER_SKILL_REQUIREMENTS.get(predicted_career, {})
        
        if not reqs:
            return 75.0, []  # No requirements mapped; assume aligned
        
        gaps = []
        total_deficit = 0.0
        
        for skill_name, min_val in reqs.items():
            # Try to get skill value from form data
            val = float(form_data.get(skill_name, 0.0))
            
            if val < min_val:
                deficit = min_val - val
                total_deficit += deficit
                gaps.append({
                    'skill': skill_name,
                    'required': min_val,
                    'current': val,
                    'gap': deficit,
                })
        
        # Calculate score: start at 100, subtract 10 points per 0.5-point gap
        alignment_score = max(50.0, 100.0 - (total_deficit * 10.0))
        
        return alignment_score, gaps

    @classmethod
    def compute_interest_alignment_score(cls, predicted_career: str) -> float:
        """
        Compute interest-career alignment score (0-100).
        
        Note: In production, would compare user_interests with career typical interests.
        Currently uses field-based heuristic.
        
        Args:
            predicted_career: Predicted career cluster
            
        Returns:
            Alignment score (0-100)
        """
        # If career is in CAREER_INTERESTS, return high score
        if predicted_career in Config.CAREER_INTERESTS:
            return 80.0  # Known career with interest profile
        
        return 70.0  # Unmapped career; neutral score

    @classmethod
    def validate_prediction(
        cls, 
        form_data: Dict, 
        predicted_career: str
    ) -> Tuple[bool, List[str], List[str], Dict]:
        """
        Validate predicted career against user background, skills, and interests.

        Args:
            form_data: Raw input dictionary with user data
            predicted_career: The career cluster predicted by the model

        Returns:
            Tuple of (
                is_aligned: bool,
                warnings: List[str],
                suggestions: List[str],
                scores: Dict with alignment scores
            )
        """
        warnings = []
        suggestions = []
        is_aligned = True
        
        # Extract features
        field = str(form_data.get('field', '')).strip()
        gpa = float(form_data.get('gpa', 0.0))
        
        # Compute alignment scores
        education_score = cls.compute_education_alignment_score(field, predicted_career)
        skill_score, skill_gaps = cls.compute_skill_alignment_score(form_data, predicted_career)
        interest_score = cls.compute_interest_alignment_score(predicted_career)
        
        # Composite alignment score (weighted average)
        composite_score = (
            0.35 * education_score +
            0.40 * skill_score +
            0.25 * interest_score
        )
        
        scores = {
            'education_alignment': education_score,
            'skill_alignment': skill_score,
            'interest_alignment': interest_score,
            'composite_alignment': composite_score,
        }
        
        # 1. Educational Field Alignment Check
        if education_score < 75.0:
            is_aligned = False
            field_lower = str(field).strip().lower()
            aligned_careers = Config.FIELD_CAREER_ALIGNMENT.get(field_lower, [])
            warnings.append(
                f"Predicted career '{predicted_career}' is not typical for "
                f"a '{field}' background."
            )
            if aligned_careers:
                suggestions.append(
                    f"Consider careers more aligned with {field}: "
                    f"{', '.join(aligned_careers[:3])}."
                )
        
        # 2. Skill Requirements Check
        if skill_gaps:
            is_aligned = False
            for gap in skill_gaps:
                skill_display = gap['skill'].replace('_', ' ').title()
                warnings.append(
                    f"Your {skill_display} ({gap['current']:.1f}) is below "
                    f"typical requirements for {predicted_career} (min {gap['required']:.1f})."
                )
                suggestions.append(
                    f"Develop {skill_display} through projects, courses, or internships."
                )
        
        # 3. GPA Check for selective careers
        selective_careers = ['Doctor & Surgeon', 'Legal Professional']
        if predicted_career in selective_careers and gpa < 3.0:
            is_aligned = False
            warnings.append(
                f"Your GPA ({gpa:.2f}) is below competitive threshold for "
                f"{predicted_career}."
            )
            suggestions.append(
                "Enhance academic record or seek professional certifications "
                "and internship credentials."
            )
        
        return is_aligned, warnings, suggestions, scores

    @classmethod
    def generate_validation_report(cls, form_data: Dict, predicted_career: str) -> Dict:
        """
        Generate comprehensive validation report with scores and recommendations.
        
        Args:
            form_data: User input data
            predicted_career: Predicted career cluster
            
        Returns:
            Dictionary with full validation details
        """
        is_aligned, warnings, suggestions, scores = cls.validate_prediction(
            form_data, predicted_career
        )
        
        field = str(form_data.get('field', '')).strip()
        skill_score, skill_gaps = cls.compute_skill_alignment_score(form_data, predicted_career)
        
        report = {
            'predicted_career': predicted_career,
            'user_field': field,
            'is_aligned': is_aligned,
            'alignment_scores': scores,
            'warnings': warnings,
            'suggestions': suggestions,
            'skill_gaps': [
                {
                    'skill': gap['skill'],
                    'gap_size': gap['gap'],
                    'current': gap['current'],
                    'required': gap['required'],
                }
                for gap in skill_gaps
            ],
        }
        
        return report
