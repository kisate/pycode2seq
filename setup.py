from setuptools import setup, find_packages

VERSION = "0.0.6"

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    install_requires = [line.strip() for line in requirements_file]

setup_args = dict(
    name="pycode2seq",
    version=VERSION,
    description="Inference and training for multiple languages of code2seq",
    long_description_content_type="text/markdown",
    long_description=readme,
    install_requires=install_requires,
    license="MIT",
    packages=find_packages(),
    author="Dmitrii Kharlapenko",
    author_email="dimkakha@gmail.com",
    keywords=["code2seq", "pytorch", "pytorch-lightning", "ml4code", "ml4se"],
    url="https://github.com/kisate/pycode2seq",
    download_url="https://pypi.org/project/pycode2seq/",
)

if __name__ == "__main__":
    setup(**setup_args)
