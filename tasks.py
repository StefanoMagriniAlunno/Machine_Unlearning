import sys

from invoke import task  # type: ignore

list_packages = [
    "jupyter",
    "sphinx sphinxcontrib-plantuml esbonio sphinx_rtd_theme",  # documentation
    "pandas",  # scientific computing
    "tqdm tabulate types-tabulate",  # utilities
    "torch torchvision --index-url https://download.pytorch.org/whl/cu124",  # gpu computing
]


@task
def download(c, cache: str):
    """download contents"""

    for pkg in list_packages:
        c.run(
            f"{sys.executable} -m pip download --no-cache-dir --dest {cache} --quiet {pkg} "
        )


@task
def install(c, cache: str):
    """Install contests"""

    for pkg in list_packages:
        c.run(
            f"{sys.executable} -m pip install --compile --no-index --find-links={cache} --quiet {pkg} "
        )
