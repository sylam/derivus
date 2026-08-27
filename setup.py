from setuptools import setup, find_packages
import os

with open('README.md') as f:
    long_description = f.read()

here = os.path.dirname(os.path.abspath(__file__))

version_ns = {}
with open(os.path.join(here, 'derivus', '_version.py')) as f:
    exec(f.read(), {}, version_ns)

# the extras compose: `desk` is the working stack of getting_started.md in one install.
# mcp-types is declared because derivus_mcp imports it directly - reaching it only through
# mcp's own dependency would be pyparsing-by-accident again
SERVICE = ['fastapi>=0.100', 'uvicorn>=0.23']
MCP = ['mcp>=2,<3', 'mcp-types', 'requests']

setup(
    name='derivus',
    version=version_ns['__version__'],
    # excel_integration is an add-in that lives in the repo, not part of the installed library.
    # derivus_mcp and derivus_bloomberg are sibling packages on the same terms as each other:
    # in the wheel, but never importing the engine (gates read their imports)
    packages=find_packages(include=['derivus', 'derivus.*', 'derivus_bloomberg', 'derivus_bloomberg.*',
                                    'derivus_mcp', 'derivus_mcp.*']),
    # the release pipeline builds web/ and copies web/dist here before `python -m build`, so a
    # wheel serves the UI it was released with; the repo tree carries neither the build nor the copy
    package_data={'derivus': ['_ui/*', '_ui/assets/*'],
                  # the seed is a QUESTIONNAIRE of unverified candidates - the evidence-refusing
                  # loader stays the trust boundary
                  'derivus_bloomberg': ['seed.json']},
    # the three console scripts are flat modules beside the package, so they need declaring
    py_modules=['derivus_bootstrap', 'derivus_batch', 'derivus_docs'],
    url='https://github.com/sylam/derivus',
    license='PolyForm-Noncommercial-1.0.0',
    # PolyForm's Notices section obliges passing the terms on with the software, so the licence
    # has to travel in the sdist and wheel rather than only live in the repo
    license_files=('LICENSE',),
    author='shuaib.osman',
    author_email='vretiel@gmail.com',
    description='An XVA quantitative library with AAD',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # torch>=2.0 does not build below 3.8, so the old >=3.6 floor let pip resolve a broken install
    python_requires='>=3.8',
    # exactly what the package imports — pyparsing was previously only reaching us by accident,
    # as a transitive dependency of matplotlib
    install_requires=['numpy>=1.16.1', 'scipy>=1.2.2', 'pandas>=1.0', 'pyparsing>=2.4.7',
                      'sortedcontainers>2.0', 'torch>=2.0'],
    extras_require={
        'interactive': ['jupyter', 'matplotlib>=3.0'],
        # GARCHSpotModel's calibration only; the import is lazy, so the rest of the library
        # installs and runs without it
        'garch': ['arch>=6.0'],
        # DV_Docs only writes mkdocs.yml; these are what building the emitted config needs
        'docs': ['mkdocs>=1.5', 'mkdocs-material>=9.0', 'pymdown-extensions>=10.0'],
        # DV_Service only; derivus/service.py is the sole importer, so `import derivus` never
        # needs either of these
        'service': SERVICE,
        # the MCP binding's own two imports (mcp needs python 3.10+); `desk` is the whole
        # working stack - service plus MCP, with the packaged UI riding in the wheel as data
        'mcp': MCP,
        'desk': SERVICE + MCP,
    },
    entry_points={
        'console_scripts': [
            'DV_Bootstrap = derivus_bootstrap:main',
            'DV_Batch = derivus_batch:main',
            'DV_Docs = derivus_docs:main',
            'DV_Service = derivus.service:main',
            'DV_MCP = derivus_mcp.server:main',
            'DV_Bloomberg = derivus_bloomberg.discover:main'
        ]},
    # PolyForm is source-available rather than OSI-approved, so Other/Proprietary is the
    # closest PyPI classifier; the SPDX identifier in `license` is the precise statement
    classifiers=['Development Status :: 4 - Beta',
                 'License :: Other/Proprietary License',
                 'Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Topic :: Office/Business :: Financial',
                 'Programming Language :: Python :: 3'],
)
