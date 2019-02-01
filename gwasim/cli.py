# -*- coding: utf-8 -*-

"""Console script for gwasim."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for gwasim."""
    click.echo("Replace this message by putting your code into "
               "gwasim.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
