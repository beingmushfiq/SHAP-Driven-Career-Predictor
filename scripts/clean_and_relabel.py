"""
Deterministic Career Relabeling Script.

Replaces random career assignment with rule-based mapping that uses
feature values (GPA, skills, experience) to select the most appropriate
career within each field group. This produces clean, learnable labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import Config

# ─── Deterministic Career Mapping Rules ─────────────────────────────
# Each field maps to a list of (career, condition_fn) tuples.
# Conditions are evaluated in order; first match wins.
# The last entry per field is the fallback (always True).

def _high_coding(row):    return row['Coding_Skills'] >= 7
def _high_analytical(row): return row['Analytical_Skills'] >= 7
def _high_problem(row):   return row['Problem_Solving_Skills'] >= 7
def _high_comm(row):      return row['Communication_Skills'] >= 7
def _high_gpa(row):       return row['GPA'] >= 3.7
def _mid_gpa(row):        return row['GPA'] >= 3.3
def _has_research(row):   return row['Research_Experience'] >= 1
def _has_internship(row): return row['Internships'] >= 2
def _high_projects(row):  return row['Projects'] >= 5
def _has_leadership(row): return row['Leadership_Positions'] >= 1
def _high_present(row):   return row['Presentation_Skills'] >= 7
def _high_network(row):   return row['Networking_Skills'] >= 7
def _has_certs(row):      return row['Industry_Certifications'] >= 1
def _high_teamwork(row):  return row['Teamwork_Skills'] >= 7
def _high_extra(row):     return row['Extracurricular_Activities'] >= 5
def _always(row):         return True


FIELD_RULES = {
    'Computer Science': [
        ('AI Researcher',          lambda r: _has_research(r) and _high_analytical(r) and _high_gpa(r)),
        ('Data Scientist',         lambda r: _high_analytical(r) and _high_problem(r)),
        ('Cybersecurity Analyst',  lambda r: _high_problem(r) and _has_certs(r)),
        ('Game Developer',         lambda r: _high_coding(r) and _high_projects(r)),
        ('Web Developer',          lambda r: _high_coding(r) and _has_internship(r)),
        ('Software Developer',     _always),
    ],
    'Engineering': [
        ('Aerospace Engineer',         lambda r: _high_gpa(r) and _has_research(r)),
        ('Biomedical Engineer',        lambda r: _high_gpa(r) and _high_analytical(r)),
        ('Chemical Engineer',          lambda r: _high_analytical(r) and _has_research(r)),
        ('Electrical Engineer',        lambda r: _high_problem(r) and _high_coding(r)),
        ('Fluid Mechanics Engineer',   lambda r: _high_analytical(r) and _high_problem(r)),
        ('Civil Engineer',             lambda r: _has_internship(r) and _high_teamwork(r)),
        ('Acoustics Specialist',       lambda r: _has_research(r)),
        ('Mechanical Engineer',        _always),
    ],
    'Medicine': [
        ('Surgeon',             lambda r: _high_gpa(r) and _high_problem(r) and _has_research(r)),
        ('Doctor',              lambda r: _high_gpa(r) and _high_comm(r)),
        ('Pharmacist',          lambda r: _high_analytical(r) and _has_certs(r)),
        ('Dentist',             lambda r: _mid_gpa(r) and _has_internship(r)),
        ('Physician Assistant', lambda r: _mid_gpa(r) and _high_teamwork(r)),
        ('Nurse',               _always),
    ],
    'Business': [
        ('Entrepreneur',               lambda r: _has_leadership(r) and _high_network(r) and _high_projects(r)),
        ('Manager',                    lambda r: _has_leadership(r) and _high_comm(r)),
        ('Construction Manager',       lambda r: _high_problem(r) and _has_internship(r)),
        ('Human Resources Specialist', _always),
    ],
    'Finance': [
        ('Investment Banker',      lambda r: _high_gpa(r) and _high_network(r) and _has_internship(r)),
        ('Actuary',                lambda r: _high_analytical(r) and _high_problem(r)),
        ('Financial Controller',   lambda r: _high_analytical(r) and _has_certs(r)),
        ('Financial Advisor',      lambda r: _high_comm(r) and _high_network(r)),
        ('Risk Analyst',           lambda r: _high_problem(r) and _has_research(r)),
        ('Credit Analyst',         lambda r: _high_analytical(r)),
        ('Financial Analyst',      lambda r: _has_internship(r)),
        ('Accountant',             _always),
    ],
    'Marketing': [
        ('Advertising Manager',         lambda r: _has_leadership(r) and _high_present(r) and _high_comm(r)),
        ('Brand Manager',               lambda r: _has_leadership(r) and _high_network(r)),
        ('Marketing Manager',           lambda r: _high_comm(r) and _has_internship(r)),
        ('Digital Marketing Specialist', lambda r: _high_coding(r) and _high_projects(r)),
        ('Social Media Manager',        lambda r: _high_network(r) and _high_extra(r)),
        ('Market Research Analyst',     lambda r: _high_analytical(r)),
        ('Marketing Specialist',        _always),
    ],
    'Law': [
        ('Judge',             lambda r: _high_gpa(r) and _has_leadership(r) and _high_comm(r)),
        ('Legal Consultant',  lambda r: _high_analytical(r) and _high_comm(r) and _has_internship(r)),
        ('Lawyer',            lambda r: _high_comm(r) and _high_problem(r)),
        ('Legal Analyst',     lambda r: _high_analytical(r) and _has_research(r)),
        ('Paralegal',         lambda r: _has_internship(r)),
        ('Legal Secretary',   _always),
    ],
    'Education': [
        ('Principal',               lambda r: _has_leadership(r) and _high_comm(r) and _high_gpa(r)),
        ('Education Administrator', lambda r: _has_leadership(r) and _high_comm(r)),
        ('Curriculum Developer',    lambda r: _high_analytical(r) and _has_research(r)),
        ('Special Education Teacher', lambda r: _high_comm(r) and _high_teamwork(r)),
        ('Teacher',                 _always),
    ],
    'Psychology': [
        ('Clinical Psychologist',                  lambda r: _high_gpa(r) and _has_research(r) and _high_analytical(r)),
        ('Forensic Psychologist',                  lambda r: _high_analytical(r) and _high_problem(r)),
        ('Industrial-Organizational Psychologist', lambda r: _has_internship(r) and _high_analytical(r)),
        ('School Psychologist',                    lambda r: _high_comm(r) and _high_teamwork(r)),
        ('Art Therapist',                          lambda r: _high_projects(r) and _high_comm(r)),
        ('Music Therapist',                        lambda r: _high_extra(r) and _high_comm(r)),
        ('School Counselor',                       lambda r: _high_comm(r)),
        ('Counselor',                              lambda r: _high_teamwork(r)),
        ('Psychologist',                           _always),
    ],
    'Biology': [
        ('Geneticist',       lambda r: _high_gpa(r) and _has_research(r) and _high_analytical(r)),
        ('Biochemist',       lambda r: _has_research(r) and _high_analytical(r)),
        ('Biotechnologist',  lambda r: _high_coding(r) and _has_research(r)),
        ('Microbiologist',   lambda r: _has_research(r) and _high_problem(r)),
        ('Ecologist',        lambda r: _high_extra(r) and _high_teamwork(r)),
        ('Zoologist',        lambda r: _high_projects(r)),
        ('Biologist',        _always),
    ],
    'Chemistry': [
        ('Analytical Chemist', lambda r: _high_analytical(r) and _has_research(r)),
        ('Organic Chemist',    lambda r: _has_research(r) and _high_gpa(r)),
        ('Physical Chemist',   lambda r: _high_problem(r) and _high_analytical(r)),
        ('Inorganic Chemist',  lambda r: _has_research(r)),
        ('Chemist',            _always),
    ],
    'Physics': [
        ('Quantum Physicist',  lambda r: _high_gpa(r) and _has_research(r) and _high_analytical(r)),
        ('Nuclear Physicist',  lambda r: _has_research(r) and _high_problem(r)),
        ('Astronomer',         lambda r: _high_analytical(r) and _high_projects(r)),
        ('Physicist',          _always),
    ],
    'Architecture': [
        ('Architect',                  lambda r: _high_gpa(r) and _high_projects(r) and _has_internship(r)),
        ('Landscape Architect',        lambda r: _high_projects(r) and _high_extra(r)),
        ('Urban Planner',              lambda r: _high_analytical(r) and _high_comm(r)),
        ('Interior Designer',          lambda r: _high_present(r) and _high_projects(r)),
        ('Architectural Technologist', _always),
    ],
    'Art': [
        ('Art Director',     lambda r: _has_leadership(r) and _high_comm(r) and _high_projects(r)),
        ('Animator',         lambda r: _high_coding(r) and _high_projects(r)),
        ('Graphic Designer', lambda r: _high_projects(r) and _has_internship(r)),
        ('Illustrator',      lambda r: _high_projects(r)),
        ('Artist',           _always),
    ],
    'Music': [
        ('Conductor',     lambda r: _has_leadership(r) and _high_comm(r) and _high_gpa(r)),
        ('Composer',       lambda r: _high_projects(r) and _has_research(r)),
        ('Sound Engineer', lambda r: _high_coding(r) and _high_analytical(r)),
        ('Music Teacher',  lambda r: _high_comm(r) and _high_present(r)),
        ('Musician',       _always),
    ],
}


def assign_career(row: pd.Series) -> str:
    """Deterministically assign a career based on field and feature values."""
    field = row['Field']
    rules = FIELD_RULES.get(field, [])
    for career, condition_fn in rules:
        if condition_fn(row):
            return career
    # Absolute fallback — should never reach here
    return rules[-1][0] if rules else 'Unknown'


def clean_and_relabel():
    """Main pipeline: load → relabel → boost features → save."""
    input_path = Config.DATA_RAW_DIR / 'career_dataset_student.csv'
    
    # Use original if available, otherwise use current
    original_path = Config.DATA_RAW_DIR / 'career_dataset_student_original.csv'
    source_path = original_path if original_path.exists() else input_path
    
    print(f"Loading {source_path}...")
    df = pd.read_csv(source_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fields: {sorted(df['Field'].unique())}")
    
    # ─── Step 1: Deterministic career assignment ────────────────────
    print("Assigning careers using deterministic rules...")
    df['Career'] = df.apply(assign_career, axis=1)
    
    # ─── Step 2: Boost field-specific features for learnable signal ─
    print("Boosting field-specific features for signal strength...")
    np.random.seed(42)
    
    boost_map = {
        'Computer Science': {'Coding_Skills': (7, 10)},
        'Engineering':      {'Analytical_Skills': (7, 10), 'Problem_Solving_Skills': (7, 10)},
        'Medicine':         {'GPA': (3.5, 4.0)},
        'Business':         {'Communication_Skills': (7, 10), 'Leadership_Positions': (1, 1)},
        'Marketing':        {'Communication_Skills': (7, 10), 'Presentation_Skills': (7, 10)},
        'Law':              {'Communication_Skills': (7, 10), 'Analytical_Skills': (6, 10)},
        'Finance':          {'Analytical_Skills': (7, 10)},
        'Education':        {'Communication_Skills': (7, 10), 'Teamwork_Skills': (6, 10)},
        'Psychology':       {'Communication_Skills': (7, 10), 'Teamwork_Skills': (7, 10)},
        'Biology':          {'Research_Experience': (1, 1), 'Analytical_Skills': (6, 10)},
        'Chemistry':        {'Research_Experience': (1, 1), 'Analytical_Skills': (7, 10)},
        'Physics':          {'Analytical_Skills': (7, 10), 'Problem_Solving_Skills': (7, 10)},
        'Architecture':     {'Projects': (4, 8), 'Presentation_Skills': (6, 10)},
        'Art':              {'Projects': (4, 8)},
        'Music':            {'Extracurricular_Activities': (5, 10)},
    }
    
    for i, row in df.iterrows():
        field = row['Field']
        if field in boost_map:
            for feat, (lo, hi) in boost_map[field].items():
                if feat == 'GPA':
                    df.at[i, feat] = round(np.random.uniform(lo, hi), 2)
                elif lo == hi:
                    df.at[i, feat] = lo
                else:
                    df.at[i, feat] = np.random.randint(lo, hi + 1)
    
    # ─── Step 3: Validate and report ────────────────────────────────
    career_counts = df['Career'].value_counts()
    print(f"\nCareer distribution ({df['Career'].nunique()} unique careers):")
    print(career_counts.to_string())
    print(f"\nMin samples per career: {career_counts.min()}")
    print(f"Max samples per career: {career_counts.max()}")
    print(f"Mean samples per career: {career_counts.mean():.1f}")
    
    # ─── Step 4: Save ──────────────────────────────────────────────
    # Backup original if not already backed up
    if not original_path.exists() and input_path.exists():
        import shutil
        shutil.copy2(input_path, original_path)
        print(f"Original dataset backed up to {original_path}")
    
    df.to_csv(input_path, index=False)
    print(f"\nDeterministic dataset saved to {input_path}")
    print("Done.")


if __name__ == '__main__':
    clean_and_relabel()
