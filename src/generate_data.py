"""
Synthetic Dataset Generator for Career Predictor.

Generates a realistic career prediction dataset with 17 features
(mix of numerical and categorical) and 15 career labels.
Uses weighted probabilities so features correlate with careers.

Run: python -m src.generate_data
"""

import numpy as np
import pandas as pd
from src.config import Config
from src.utils import set_seeds

set_seeds(Config.RANDOM_SEED)


def generate_career_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic career prediction dataset matching the current schema."""
    
    # Define fields and their associated careers for logical generation
    field_mapping = {
        'Computer Science': ['Software Developer', 'Cybersecurity Analyst', 'Data Scientist', 'Web Developer', 'Game Developer', 'AI Researcher'],
        'Engineering': ['Civil Engineer', 'Mechanical Engineer', 'Electrical Engineer', 'Chemical Engineer', 'Acoustics Specialist', 'Aerospace Engineer', 'Biomedical Engineer', 'Fluid Mechanics Engineer'],
        'Medicine': ['Surgeon', 'Doctor', 'Nurse', 'Pharmacist', 'Dentist', 'Physician Assistant'],
        'Business': ['Manager', 'Entrepreneur', 'Human Resources Specialist', 'Construction Manager'],
        'Finance': ['Accountant', 'Financial Analyst', 'Investment Banker', 'Actuary', 'Financial Controller', 'Financial Advisor', 'Credit Analyst', 'Risk Analyst'],
        'Marketing': ['Marketing Manager', 'Social Media Manager', 'Brand Manager', 'Market Research Analyst', 'Advertising Manager', 'Marketing Specialist', 'Digital Marketing Specialist'],
        'Law': ['Lawyer', 'Legal Consultant', 'Judge', 'Paralegal', 'Legal Analyst', 'Legal Secretary'],
        'Education': ['Teacher', 'Principal', 'Education Administrator', 'Special Education Teacher', 'Curriculum Developer'],
        'Psychology': ['Psychologist', 'Counselor', 'School Psychologist', 'Clinical Psychologist', 'Art Therapist', 'Music Therapist', 'Forensic Psychologist', 'Industrial-Organizational Psychologist', 'School Counselor'],
        'Biology': ['Biologist', 'Microbiologist', 'Geneticist', 'Biochemist', 'Biotechnologist', 'Ecologist', 'Zoologist'],
        'Chemistry': ['Chemist', 'Organic Chemist', 'Analytical Chemist', 'Inorganic Chemist', 'Physical Chemist'],
        'Physics': ['Physicist', 'Astronomer', 'Nuclear Physicist', 'Quantum Physicist'],
        'Architecture': ['Architect', 'Urban Planner', 'Interior Designer', 'Landscape Architect', 'Architectural Technologist'],
        'Art': ['Art Director', 'Graphic Designer', 'Artist', 'Illustrator', 'Animator'],
        'Music': ['Conductor', 'Music Teacher', 'Composer', 'Musician', 'Sound Engineer']
    }

    fields = list(field_mapping.keys())
    data_list = []

    for _ in range(n_samples):
        # 1. Pick a Field
        field = np.random.choice(fields)
        
        # 2. Pick a Career from that field
        career = np.random.choice(field_mapping[field])
        
        # 3. Generate Features based on field logic
        row = {
            'Field': field,
            'Career': career,
            'GPA': np.round(np.random.uniform(2.5, 4.0), 2),
            'Extracurricular_Activities': np.random.randint(0, 11),
            'Internships': np.random.randint(0, 6),
            'Projects': np.random.randint(0, 11),
            'Leadership_Positions': np.random.choice([0, 1], p=[0.7, 0.3]),
            'Field_Specific_Courses': np.random.randint(2, 11),
            'Research_Experience': np.random.choice([0, 1], p=[0.8, 0.2]),
            'Coding_Skills': np.random.randint(0, 6),
            'Communication_Skills': np.random.randint(0, 6),
            'Problem_Solving_Skills': np.random.randint(0, 6),
            'Teamwork_Skills': np.random.randint(0, 6),
            'Analytical_Skills': np.random.randint(0, 6),
            'Presentation_Skills': np.random.randint(0, 6),
            'Networking_Skills': np.random.randint(0, 6),
            'Industry_Certifications': np.random.choice([0, 1], p=[0.75, 0.25]),
        }

        # Boost relevant features based on field to make the data learnable
        if field == 'Computer Science':
            row['Coding_Skills'] = np.random.randint(3, 6)
            row['Problem_Solving_Skills'] = np.random.randint(3, 6)
        elif field == 'Medicine':
            row['GPA'] = np.round(np.random.uniform(3.4, 4.0), 2)
            row['Analytical_Skills'] = np.random.randint(3, 6)
        elif field in ['Finance', 'Engineering']:
            row['Analytical_Skills'] = np.random.randint(3, 6)
            row['Problem_Solving_Skills'] = np.random.randint(3, 6)
        elif field in ['Marketing', 'Business', 'Law']:
            row['Communication_Skills'] = np.random.randint(3, 6)
            row['Presentation_Skills'] = np.random.randint(3, 6)
        elif field == 'Art':
            row['Projects'] = np.random.randint(4, 11)

        data_list.append(row)

    df = pd.DataFrame(data_list)
    return df


if __name__ == '__main__':
    Config.ensure_directories()
    df = generate_career_data(1000)
    
    # Updated output path to career_dataset_student.csv as requested
    path = Config.DATA_RAW_DIR / 'career_dataset_student.csv'
    df.to_csv(path, index=False)
    
    print(f"Dataset generated successfully at: {path}")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nField distribution:\n{df['Field'].value_counts().head()}")

