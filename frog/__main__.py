import typer

import frog.doe.cli

app = typer.Typer()

app.add_typer(frog.doe.app, name='doe')

if __name__ == "__main__":
    app()