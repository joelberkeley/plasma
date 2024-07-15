import pandas as pd
import numpy as np
import plotly.graph_objs as go

p = pd.read_csv("ions.csv")
e = pd.read_csv("electrons.csv")

p = [x.drop("p", axis="columns") for _, x in p.groupby("p")]
e = [x.drop("p", axis="columns") for _, x in e.groupby("p")]

fig = go.Figure()

for colour, trace in [("blue", p), ("purple", e)]:
    for particle, t in enumerate(trace):
        t = np.array(t)
        fig.add_trace(
            go.Scatter3d(
                x=t[:, 0],
                y=t[:, 1],
                z=t[:, 2],
                mode="lines",
                marker={"color": colour},
            )
        )

fig.write_html("index.html")
