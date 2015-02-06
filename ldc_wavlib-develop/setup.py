import os;
import subprocess;


# we only need setuptools for development for
# access to develop command
try:
    from setuptools import setup
except ImportError:
    from distutils import setup;
from Cython.Build import cythonize;


# Take care of version naming here.
# Must change VERSION each time we make a new tag.
# Code taken from numpy/setup.py
VERSION = '0.9.6';

def get_version_info():
    """
    """
    if os.path.exists('.git'):
        git_revision = git_version();
        full_version = VERSION +  '.dev-' + git_revision[:7];
    else:
        full_version = VERSION;
    return full_version;


def git_version():
    """
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {};
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k);
            if v is not None:
                env[k] = v;
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C';
        env['LANG'] = 'C';
        env['LC_ALL'] = 'C;'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0];
        return out;

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD']);
        git_revision = out.strip().decode('ascii');
    except OSError:
        git_revision = "Unknown";

    return git_revision;


def write_version_py(fn, version):
    """
    """
    with open(fn, 'w') as f:
        f.write("version = '%s'\n" % version);


version = get_version_info();
write_version_py(os.path.join('ldc_wavlib', 'version.py'), version);

setup(
    name = "ldc_wavlib",
    version = version,
    author = "Neville Ryant",
    author_email = "nryant@gmail.com",
    description = ("Feature extraction for audio."),
    license = "BSD",
    packages=['ldc_wavlib', 'ldc_wavlib.features'],
    ext_modules = cythonize('ldc_wavlib/features/*.pyx')
);
