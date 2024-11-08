#imports
import numpy as np
import pandas as pd
from mpi4py import MPI
from dolfinx.plot import vtk_mesh
import ufl

from dolfinx import fem, mesh, io,geometry
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import Constant
from ufl import  dx, grad, inner,  Measure, lhs ,rhs
import dolfinx
from dolfinx import default_scalar_type
from dolfinx.mesh import locate_entities, meshtags
from dolfinx.io import XDMFFile




# parameter values 
np.random.seed(42)
k_values = [np.random.uniform(10, 200) for _ in range(1000)]

#Consider the domain as a Unit square [0,1] X [0,1]
domain =  mesh.create_unit_square(MPI.COMM_WORLD, 51, 51)
V = fem.functionspace(domain, ("Lagrange", 1))

#definijg the boundaries of the domain- marked 1 for bottom edge and 2 for remianing edges
boundaries = [(2, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1)),
              (1, lambda x: np.isclose(x[1], 0)),
              (2, lambda x: np.isclose(x[1], 1))]

#Facets on the boundary and marked with marker for custom integration measure ds
facet_indices, facet_markers = [], []
fdim = domain.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(domain, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])


#custom integration measure for neumann and robin conditons
ds = Measure("ds", domain=domain, subdomain_data=facet_tag)

T = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)

#defining the boundary conditions using the variational problem. For example, ds(1)- indicates the integration of function only on the bottom edge using neumann conditions
class BoundaryCondition():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Robin":
                self._bc = 0.5 * inner(T, v) * ds(marker)
        elif type == "Neumann":
            self._bc = -1*inner(1, v)* ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc

    @property
    def type(self):
        return self._type

# Define the Dirichlet condition
boundary_conditions = [BoundaryCondition("Robin", 2,fem.Constant(domain, default_scalar_type(0.5))),
                       BoundaryCondition("Neumann", 1, fem.Constant(domain, default_scalar_type(-1)))]



#running a for loop for U_vales geneterated to evaluate temperature values for each heat conductivity value
#getting temperature values at 10 different points in the domain(needed to add Z axis also but remains to be 0(2D problem)
#motivated from https://jsdokken.com/dolfinx-tutorial/chapter1/membrane_code.html

points = np.zeros((3, 10))
x = [round(np.random.uniform(0, 1), 2) for _ in range(10)]
y = [round(np.random.uniform(0, 1), 2) for _ in range(10)]
points[0]=x
points[1]=y
temperature_values = []
for value in positive_samples:
    kappa = fem.Constant(domain, default_scalar_type(value))
    F = kappa * inner(grad(T), grad(v)) * dx

    # Appending Dirichlet conditions if any to bcs and adding remaining boundary conditions to the variational problem F
    bcs = []
    for condition in boundary_conditions:
        if condition.type == "Dirichlet":
            bcs.append(condition.bc)
        else:
            F += condition.bc

    # Separating a(terms depending on (T,v) as LHS of F and L(terms depending on v) as RHS of F
    a = lhs(F)
    L = rhs(F)

    # Solving the linear problem
    problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly",  "pc_type": "lu"})
    Th = problem.solve()


    #creating a bounding box tree of the cells of the mesh, which allows a quick recursive search through the mesh entities.
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []##the points found on the current processor.

    # Find cells whose bounding-box collide with the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)

    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i))

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    cells  = np.concatenate(cells)#flattening
    #evaluating T values using .eval() function
    T_values = Th.eval(points_on_proc, cells)
    temperature_values.append(T_values)


# field names
fields = ['x1', 'x2', 'x3', 'x4','x5','x6','x7','x8','x9','x10']
data = pd.DataFrame(temperature_values, columns = fields)

data['k'] = k_values
temp_data = data.applymap(lambda x: x.strip('[]'))
data.to_csv('temp_data_2DSS.csv',index=False)


