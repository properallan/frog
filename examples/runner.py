from backnozzle.cad import BackNozzleCAD
from pyqode.utils import radius_to_area
from pyqode.solver import Solver, SU2Solver

def run_1D(config):
    nozzle = BackNozzleCAD(
        divergence_angle=config['divergence_angle'],
        throat_radius=config['throat_radius'],
    )

    points = nozzle.get_points(config['domain_size'])
    xn, rn = points[:, 0], points[:, 1]

    config['domain_x'] = xn
    config['domain_area'] = radius_to_area(rn),

    q1d = Solver(config = config)
    q1d.run()

    return q1d.check_convergence()

def run_2D(config):
    nozzle = BackNozzleCAD(
        divergence_angle=config['divergence_angle'],
        throat_radius=config['throat_radius'],
    )

    points = nozzle.get_points(config['domain_Nx'])
    xn, rn = points[:, 0], points[:, 1]

    config['domain_x'] = xn
    config['domain_area'] = radius_to_area(rn)

    su2 = SU2Solver(config=config)
    su2.run()

    #return su2.check_convergence()
    return True
