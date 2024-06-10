from setuptools import setup, find_packages

def requirements_to_list()->list:
    requirements = []
    with open('requirements.txt', 'r') as f:
        for line in f:
            requirements.append(line.strip())
    return requirements

def get_readme() -> str:
    with open('README.md', 'r') as f:
        return f.read()

readme_content = get_readme()

setup(
    name='unmixing',
    version='0.2.0',
    packages=find_packages(),
    install_requires = [
        'torch',
        'numpy'
    ],
    long_description=readme_content,
    long_description_content_type='text/markdown',
)