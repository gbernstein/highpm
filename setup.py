import sys,os,glob,re
import select

try:
    from setuptools import setup
    import setuptools
    print("Using setuptools version",setuptools.__version__)
except ImportError:
    from distutils.core import setup
    import distutils
    print("Using distutils version",distutils.__version__)

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

dependencies = ['numpy', 'astropy', 'easyaccess', 'pixmappy']

with open('README.md') as file:
    long_description = file.read()

# Read in the version 
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file=os.path.join('highpm','_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    highpm_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('HIGHPM version is %s'%(highpm_version))

data = glob.glob(os.path.join('data','*'))

dist = setup(
        name="HIGHPM",
        version=highpm_version,
        author="Gary Bernstein",
        author_email="garyb@PHYSICS.UPENN.EDU",
        description="Search for high-proper-motion stars in DES",
        long_description=long_description,
        license = "GPL License",
        url="https://github.com/gbernstein/highpm",
        download_url="https://github.com/gbernstein/highpm/releases/tag/v%s.zip"%highpm_version,
        packages=['highpm'],
        install_requires=dependencies,
    )

