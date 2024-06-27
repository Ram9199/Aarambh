from setuptools import setup, find_packages

setup(
    name='aarambh',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'torch',
        'transformers',
        'numpy',
        'pandas',
        'opencv-python',
        'pytesseract',
        'SpeechRecognition',
        'pydub',
    ],
)
