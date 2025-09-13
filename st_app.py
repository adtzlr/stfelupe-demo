import streamlit as st
from stpyvista.trame_backend import stpyvista
from stpyvista.utils import start_xvfb
import felupe as fem
import numpy as np

start_xvfb()

st.set_page_config(layout="wide")

st.sidebar.title("FElupe")
n = st.sidebar.slider("Details", 2, 11, 4)
v = st.sidebar.slider("Stretch", 1.0, 2.0, 2.0)

progress_bar = st.progress(0, text="Progress")
def show_progress(i, j, substep):
    progress_bar.progress((1 + j) / len(move))

H = st.sidebar.slider("Height", 10.0, 500.0, 50.0)  # height in mm
D = st.sidebar.slider("Diameter", 10.0, 500.0, 100.0)  # diameter in mm
d = st.sidebar.slider("Hole", 0.0, 500.0, 50.0)  # diameter in mm
n = st.sidebar.slider("Repetitions", 1, 8, 3)  # number of rubber layers
t = st.sidebar.slider("Separator height", 1.0, 6.0, 3.0, 0.25)  # thickness of metal sheet in mm

axial_max = st.sidebar.slider("Max. deformation", 0.0, 1.0, 0.2)  # max. axial deformation
axial_steps = st.sidebar.slider("Steps", 1, 50, 10)  # number of axial steps

mu = 1.0  # shear modulus in MPa
K = 5000.0  # bulk modulus in MPa

surface = 2  # 0 = rectangle, 1 = outside, 2 = inside and outside
details = 6  # number of points per layer
revolve = 12  # number of elements for revolution (180°)

plot_mesh = True
plot_curve_axial = True
plot_stiffness_lateral = True
plot_model_axial = True
plot_model_lateral = False

# %% model setup
R = D / 2
r = d / 2
h = H / n - t * (n - 1)

number_of_points = details, int((R - r) / h) * details
rect = fem.Rectangle(a=(0, r), b=(h, R), n=number_of_points)

if surface:
    rect = rect.add_runouts(
        [0.1 * surface],
        centerpoint=[h / 2, r + (surface - 1) * (R - r) / 2],
        normalize=True,
    )

sheet = fem.Rectangle(a=(h, r), b=(h + t, R), n=(2, number_of_points[1]))

values = fem.math.linsteps([0, n * (h + t)], num=n)
list_of_rects = [rect.translate(v, axis=0) for v in values]
list_of_sheets = [sheet.translate(v, axis=0) for v in values[:-1]]

rects = fem.MeshContainer(list_of_rects, merge=True).stack()
sheets = fem.MeshContainer(list_of_sheets, merge=True).stack()

meshes = fem.MeshContainer([rects, sheets], merge=True, decimals=3)

if plot_mesh:
    meshes.plot(colors=[None, "white"]).show()

regions = [fem.RegionQuad(m) for m in meshes]
field_rubber, field_metal = [fem.FieldAxisymmetric(r) for r in regions]

fields, field = (field_rubber & field_metal).merge()
umats = [
    fem.NeoHooke(mu=mu),
    fem.LinearElasticLargeStrain(E=2.1e5, nu=0.3),
]
solids = [
    fem.SolidBodyNearlyIncompressible(umat=umats[0], field=fields[0], bulk=K),
    fem.SolidBody(umat=umats[1], field=fields[1]),
]

move = fem.math.linsteps([0, 1], num=axial_steps) * n * h * -axial_max

boundaries, loadcase = fem.dof.uniaxial(field, clamped=True, sym=False)
ramp = {boundaries["move"]: move}
step = fem.Step(items=solids, ramp=ramp, boundaries=boundaries)

stiffness_lateral = []


def shear_stiffness(i, j, substep):

    solids_3d = [s.revolve(n=revolve + 1, phi=180) for s in solids]
    field_3d = field.revolve(n=revolve + 1, phi=180)

    boundaries_3d, loadcase = fem.dof.shear(
        field_3d, axes=(1, 0), sym=(0, 0, 1), moves=(0, 0, move[j])
    )
    move_3d = fem.math.linsteps([1], num=0) * n * h * 1e-6
    ramp_3d = {boundaries_3d["move"]: move_3d}
    step_3d = fem.Step(items=solids_3d, boundaries=boundaries_3d, ramp=ramp_3d)
    job_3d = fem.CharacteristicCurve(steps=[step_3d], boundary=boundaries_3d["move"])
    job_3d.evaluate(
        x0=field_3d, tol=1e-1, verbose=False, parallel=True
    )

    stiffness_lateral.append(job_3d.y[0][1] * 2 * 1e6)

    if plot_model_lateral:
        plotter = solids_3d[1].plot(
            show_undeformed=False, color="white", show_edges=False
        )
        plotter = solids_3d[0].plot(
            "Principal Values of Cauchy Stress",
            plotter=plotter,
            show_undeformed=False,
            project=fem.topoints,
        )
        stpyvista.plot(plotter)


job = fem.CharacteristicCurve(
    steps=[step], boundary=boundaries["move"], callback=shear_stiffness
).evaluate(x0=field, tol=1e-2, verbose=1)

if plot_curve_axial:
    fig, ax = job.plot(
        xaxis=0,
        yaxis=0,
        xscale=-1,
        yscale=-0.001,
        xlabel=r"Displacement $d_1$ in mm $\longrightarrow$",
        ylabel=r"Normal Force $F_1$ in kN $\longrightarrow$",
    )
    st.pyplot(fig)

if plot_stiffness_lateral:
    fig, ax = job.plot(
        y=np.array(stiffness_lateral).reshape(-1, 1),
        xaxis=0,
        yaxis=0,
        xscale=-1,
        yscale=0.001,
        xlabel=r"Displacement $d_1$ in mm $\longrightarrow$",
        ylabel=r"Shear Stiffness $k_2$ in kN / mm $\longrightarrow$",
    )
    st.pyplot(fig)
    force = job.y.copy()
    fig, ax = job.plot(
        x=force,
        y=np.array(stiffness_lateral).reshape(-1, 1),
        xaxis=0,
        yaxis=0,
        xscale=-0.001,
        yscale=0.001,
        xlabel=r"Normal Force $F_1$ in kN $\longrightarrow$",
        ylabel=r"Shear Stiffness $k_2$ in kN / mm $\longrightarrow$",
    )
    st.pyplot(fig)

if plot_model_axial:
    plotter = solids[1].plot(show_undeformed=False, color="white", show_edges=False)
    plotter = solids[0].plot(
        "Principal Values of Cauchy Stress",
        plotter=plotter,
        show_undeformed=False,
        project=fem.topoints,
    )
    stpyvista.plot(plotter)

st.toast("Simulation finished!", icon="🎉")