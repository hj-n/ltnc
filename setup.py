import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name="lsnc",
	version="0.0.1",
	author="Hyeon Jeon",
	author_email="hj@hcil.snu.ac.kr",
	description="Implementation of Label-Stretching and Label-Compression", 
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/hj-n/lsnc",
	classifiers=[
			"Programming Language :: Python :: 3",
			"License :: OSI Approved :: MIT License",
			"Operating System :: OS Independent",
	],
	install_requires=[
		"numpy",
		"btwim"
	],
	package_dir={"": "src"},
	packages=setuptools.find_packages(where="src"),
	python_requires=">=3.9.0",
)