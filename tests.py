import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

N = 100
x = [0, 1, 2, 3, 5]
y = ["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01"]
fig = make_subplots(
    2,
    1,
    row_heights=[0.98, 0.02],
    vertical_spacing=0.1,
)
for i in range(1, 3):
    for j in range(1, 2):
        if i == 1:
            fig.append_trace(go.Scatter(x=x, y=np.random.random(N)), i, j)
        if i == 2:
            fig.append_trace(go.Scatter(x=y, y=[0, 0, 0, 0, 0]), i, j)
fig.update_xaxes(matches="x")
fig.show()