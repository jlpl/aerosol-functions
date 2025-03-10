import subprocess 
import os

def run_app():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(current_dir, 'aerosol_analyzer.py')
    subprocess.run(['bokeh', 'serve', '--show', app_dir])

if __name__ == '__main__':
    run_app()

