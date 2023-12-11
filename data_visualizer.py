import plotly.express as px
from sklearn.decomposition import PCA
from data_loader import load_dataset1, load_dataset2


def visualize_data(set=1):
    if set == 2:
        x, y = load_dataset2()
    else:
        x, y = load_dataset1()
    pca = PCA(n_components=3)

    components = pca.fit_transform(x)

    fig = px.scatter_3d(
        components,
        x=0,
        y=1,
        z=2,
        color=y,
        labels={"0": "PC 1", "1": "PC 2", "2": "PC 3"},
    )
    fig.show()


if __name__ == "__main__":
    visualize_data(2)
