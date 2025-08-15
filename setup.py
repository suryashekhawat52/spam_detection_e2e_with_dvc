from setuptools import setup, find_packages
from typing import List

HYPEN_DOT_E = '-e .'
def get_requirements(file_path:str) -> List[str]:
    requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        requirement = [req.replace("\n","") for req in requirement]

    if HYPEN_DOT_E in requirement:
        requirement.remove(HYPEN_DOT_E)

    return requirement

setup(
    name = 'Spam Detection Model',
    version= '0.0.1',
    author = 'Surya',
    author_email= 'suryashekhawat52@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages=find_packages()
)