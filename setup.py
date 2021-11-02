import os
import re
import sys
import sysconfig
import site
import platform
import subprocess
try:
    import pathlib
except ImportError:
    raise ImportError("QLens requires the 'pathlib' package for installation. Please install it and try again.")

import multiprocessing

from distutils.version import LooseVersion
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext_orig):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            raise RuntimeError("Sorry, QLens doesn't work on Windows platforms. Please use Linux or OSX.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        try:
           build_temp.mkdir(parents=True)
        except FileExistsError:
           pass # This is not a problem, but exist_ok argument doesn't exist in Python <3.5
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        try:
           extdir.mkdir(parents=True)
        except FileExistsError:
           pass

        # Temporary directory where libraries and other built files should go
        # These will get automatically copied to the final install location
        print('extdir:', extdir)
        #quit()
        libout = str(extdir.parent.absolute()) + '/qlens'

        # Set "ORIGIN" variable depending on OS
        #if    platform.system() == "Linux":
            #origin = r"$ORIGIN"
        #elif  platform.system() == "Darwin": 
            ## This is any OSX-like system
            #origin = r"@loader_path"
        #else:
            #raise RuntimeError("Unrecognised operating system! pyScannerBit is only compatible with Linux and OSX-like operating systems! Aborting install...")

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + libout,
                      '-DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF',
                      '-Wno-dev',
                      '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + libout,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                    ]

        #self.debug = True
        #cfg = 'Debug' if self.debug else 'Release'
        #build_args = ['--config', cfg]

        ncpus = multiprocessing.cpu_count()
        if ncpus>1:
           ncpus -= 1 # Use 1 fewer cpus than available, so the OS can still do other things
       
        #cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args = ['--', '-j{0}'.format(ncpus)]

        env = os.environ.copy()
        #env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
        #                                                      self.distribution.get_version())
       
        # # Debugging paths
        # print("Debugging paths:")
        print("extdir:",extdir)
        print("libout:",libout)
        subprocess.check_call(['pwd'], cwd=self.build_temp, env=env)
        subprocess.check_call(['echo',self.build_temp], cwd=self.build_temp, env=env)
        # # cwd = pathlib.Path().absolute()
        # # print("ext.sourcedir:")
        # # subprocess.check_call(['echo', ext.sourcedir], cwd=self.build_temp, env=env)
        # # subprocess.check_call(['echo',cwd], cwd=self.build_temp, env=env) 
        subprocess.check_call(['ls', ext.sourcedir], cwd=self.build_temp, env=env)
        #subprocess.check_call(['ls', ext.sourcedir+'/qlens-beta'], cwd=self.build_temp, env=env)
   
        # untar ScannerBit tarball
        #subprocess.check_call(['tar','-C','pyscannerbit/scannerbit/untar/ScannerBit','-xf','pyscannerbit/scannerbit/ScannerBit_stripped.tar','--strip-components=1'], cwd=ext.sourcedir, env=env)
      
        # First cmake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=str(build_temp), env=env)
        # Main build
        subprocess.check_call(['cmake', '--build', '.', '--target', 'deps'] + build_args, cwd=str(build_temp))
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=str(build_temp), env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'python_install'] + build_args, cwd=str(build_temp))
        # Install
        #subprocess.check_call(['cmake', '--build', '.', '--target', 'install'], cwd=str(build_temp))

        print("Checking contents of temporary directory {0}".format(libout))
        print(subprocess.check_call(['ls', libout], cwd=self.build_temp, env=env))
 
setup(
    name='qlens-beta',
    version='0.0.1',
    author='Quinn Minor',
    # Add yourself if you contribute to this package
    author_email='qminor@gmail.com',
    description='Awesome lensing software.',
    long_description='',
    ext_modules=[CMakeExtension('_qlens')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=[''],
    install_requires=[
        'mpi4py',
        'matplotlib',
    ],
    extras_require = {
    }
)
