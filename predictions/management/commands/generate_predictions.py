# predictions/management/commands/generate_predictions.py
from django.core.management.base import BaseCommand
from predictions.train_model import EnvironmentalPredictor

class Command(BaseCommand):
    help = 'Generate environmental predictions'
    
    def add_arguments(self, parser):
        parser.add_argument('--hours', type=int, default=12, help='Hours ahead to predict')
    
    def handle(self, *args, **options):
        hours = options['hours']
        
        predictor = EnvironmentalPredictor()
        predictor.generate_predictions(hours_ahead=hours)
        
        self.stdout.write(self.style.SUCCESS(f'Generated predictions for next {hours} hours'))