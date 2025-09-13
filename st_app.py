import streamlit as st
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb
import felupe as fem

# Initial configuration
start_xvfb()

import pyvista as pv

# plotter = pv.Plotter()

st.title("A Streamlit app for FElupe")
n = st.sidebar.slider("Details", 2, 11, 4)
v = st.sidebar.slider("Stretch", 1.0, 2.0, 2.0)

progress_bar = st.sidebar.progress(0)
def show_progress(i, j, substep):
    progress_bar.progress((1 + j) / len(move))

mesh = fem.Cube(n=n)
region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)
umat = fem.NeoHooke(mu=1, bulk=2)
solid = fem.SolidBody(umat=umat, field=field)
move = fem.math.linsteps([0, v - 1], num=5)

ramp = {boundaries["move"]: move}
step = fem.Step(items=[solid], ramp=ramp, boundaries=boundaries)
job = fem.Job(steps=[step], callback=show_progress)
job.evaluate(tol=1e-2)

plotter = solid.plot("Principal Values of Cauchy Stress")
stpyvista(plotter)