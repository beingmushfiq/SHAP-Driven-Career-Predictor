"""
WSGI config for career_predictor project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.0/howto/deployment/wsgi/
"""

import os
import sys
from pathlib import Path
from django.core.wsgi import get_wsgi_application

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'career_predictor.settings')

application = get_wsgi_application()
