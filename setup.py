from setuptools import setup, find_packages

setup(
    name='EEG-Diff',  # Replace with your own package name
    version='0.1.0',  # The initial release version
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    description='A brief description of the EEGM project',  # Provide a short description
    long_description=open('README.md').read(),  # This will read your README file to use as the long description
    long_description_content_type='text/markdown',  # This is the format of your README file
    url='https://github.com/yourusername/EEG-Diff',  # Replace with the URL to your project repository
    packages=find_packages(),  # Automatically find and include all packages
    classifiers=[
        # Classifiers help users find your project by categorizing it.

        # Project maturity
        'Development Status :: 3 - Alpha',

        # Intended audience
        'Intended Audience :: Developers',

        # Topic
        'Topic :: Software Development :: Libraries',

        # License (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',  # Your package's Python version compatibility
    entry_points={
        'console_scripts': [
            'dvm=dvm.cli:main',  # This allows users to run 'dvm' command in the terminal.
        ],
    },
    include_package_data=True,  # This tells setuptools to include any data files it finds in your packages.
    license='MIT',  # Your project's license
    keywords='EEG, Diffusion, Brain, Signal Processing',  # Keywords to find your project
    zip_safe=False,  # This tells setuptools to not package your project as a .egg file
)
