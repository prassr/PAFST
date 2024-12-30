from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

with open("requirements.txt", 'r', encoding='utf-8') as f:
    requirements = f.read()

setup(
    name='pafst',
    version='1.0',
    author='ashtavakra',
    author_email="vidyaaltar@gmail.com",
    description='Library That Preprocessing Audio For TTS/STT.',
    install_requires=requirements,
    license='MIT',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/prassr/PAFST',
    python_requires=">=3.10, <3.11",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.10',
    ],
    keywords='speechrecognition asr voiceactivitydetection vad webrtc pafst audio denoising speaker diarization',
    py_modules=['pafst']

)
