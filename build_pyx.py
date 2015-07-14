__author__ = 'humblercoder'

if __name__ == '__main__':
    from Cython.Build import cythonize
    from distutils.core import Extension, Distribution
    from distutils.command.build_ext import build_ext
    import numpy
    extensions = [
                    Extension("_peak_detection", ["_peak_detection.pyx"],
                              include_dirs=[numpy.get_include()],
                              language="c++")
                ]
    dist = Distribution({'name': '_peak_detection', 'ext_modules': cythonize(extensions)})
    cmd = build_ext(dist)
    cmd.build_lib = "."
    cmd.ensure_finalized()
    cmd.run()