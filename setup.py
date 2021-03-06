from setuptools import setup

setup(
   name="massinference",
   version="0.0.0",
   author="Phil Marshall, Tom Collett, Spencer Everett, David Thomas",
   author_email="pjm@slac.stanford.edu, thomas.collett@port.ac.uk, spencerweverett@gmail.com, "
                "dthomas5@stanford.edu",
   url="https://github.com/davidthomas5412/massinference",
   packages=["massinference"],
   description="Line of sight weak lensing and mass inference.",
   long_description=open("README.md").read(),
   package_data={"": ["README.md", "LICENSE"]},
   include_package_data=True,
   classifiers=[
       "Development Status :: 4 - Beta",
       "License :: OSI Approved :: MIT License",
       "Intended Audience :: Developers",
       "Intended Audience :: Science/Research",
       "Operating System :: OS Independent",
       "Programming Language :: Python",
   ],
   install_requires=["numpy",  "requests"],
)