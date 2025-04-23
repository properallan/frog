import typer

import frog.doe.cli
import frog.datahandler.cli
import frog.dimensionality_reduction.cli
import frog.neuralnetwork.cli
import frog.gaussian_process.cli_wrong

app = typer.Typer()

app.add_typer(frog.doe.app, name='doe')
app.add_typer(frog.datahandler.app, name='dataset')
app.add_typer(frog.dimensionality_reduction.app, name='dr')
app.add_typer(frog.neuralnetwork.app, name='nn')
app.add_typer(frog.gaussian_process.app, name='gp')


if __name__ == "__main__":
    app()