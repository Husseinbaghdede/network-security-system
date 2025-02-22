from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    This function reads the requirements.txt file and returns a list of requirements.
    It is used to dynamically include all the required packages in the setup function.
    """
    requirement_list:List[str] = []
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                requirement = line.strip()
                ##ignore empty lines and -e . 
                if requirement and requirement!= '-e .':
                    requirement_list.append(requirement) 
    except FileNotFoundError:
        print("requirements.txt file not found.")
    
    return requirement_list


setup(
    name='NetworkSecurity',
    version='0.0.1',
    description='Network Security Project',
    author='Hussein Baghdadi',
    author_email='hussein.baghdadi01@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)