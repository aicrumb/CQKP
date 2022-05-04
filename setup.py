import setuptools

setuptools.setup(
    name='cqkp',
    version='0.0.1',
    author='Chloe R (aicrumb)',
    author_email='aicrumbmail@gmail.com',
    description='Contrastive Question-Knowledge Pretraining',
    long_description='Using two embedding models, this packages main use is finding an article which contains the answer to a question',
    long_description_content_type="text/markdown",
    url='https://github.com/aicrumb/CQKP',
    project_urls = {
        "Bug Tracker": "https://github.com/aicrumb/CQKP/issues"
    },
    license='GNU GPLv3',
    packages=['cqkp'],
    install_requires=['wandb', 'transformers', 'pytorch'],
)