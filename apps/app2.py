import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bio as dashbio

from app import app

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pickle


# 데이터 불러오기 (전역변수처럼)
df = pd.read_csv("./apps/TCGA_GTEX_SLC_103.csv")
file_ = open("./apps/slc_cancer_biomarker_dictionary", "rb")
content = pickle.load(file_)
df2 = pd.read_csv("./apps/cancer_survival_time.csv")

# 레이아웃
layout = html.Div(
    [
        # navigation
        html.Div(
            [
                html.Div(
                    html.A(
                        html.Img(src="/assets/nav_home.png"),
                        href="/",
                    ),
                    className="nav-con",
                ),
                html.Div(
                    html.A(
                        html.Img(src="/assets/nav_page1.png"),
                        href="/apps/app1",
                    ),
                    className="nav-con",
                ),
                html.Div(
                    html.A(
                        html.Img(src="/assets/nav_page2.png"),
                        href="/apps/app2",
                    ),
                    className="nav-con",
                ),
                html.Div(
                    html.A(
                        html.Img(src="/assets/nav_page3.png"),
                        href="/apps/app3",
                    ),
                    className="nav-con",
                ),
            ],
            style={
                "display": "flex",
                "flex-direction": "row",
                "justify-content": "center",
                "align-items": "center",
                "margin": "0 auto",
                "width": "80%",
                "height": "15vh",
            },
        ),
        # block 1 (위쪽) 상단배치 완료 -> pca&lda 고정
        html.Div(
            [
                html.H2("Visualization"),
                html.Br(),
                # pca
                html.Div(
                    [
                        html.H3("PCA(Component=2)"),
                        dcc.Graph(id="draw-PCA", style={"height": "65vh"}),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "float": "left",
                        "textAlign": "center",
                    },  # block2 내 왼쪽
                ),
                # lda
                html.Div(
                    [
                        html.H3("t-SNE(Component=2)"),
                        dcc.Graph(id="draw-TSNE", style={"height": "65vh"}),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "float:": "right",  # block2 내 오른쪽
                        "textAlign": "center",
                    },
                ),
                html.Br(),
            ],
            style={"width": "100%", "display": "inline-block", "textAlign": "center"},
        ),
        # block 2 (왼쪽) -> cancer 선택에 따라 보여지는 slc가 달라지도록 수정 완료
        html.Div(
            [
                html.Div(
                    [
                        html.H3("  Cancer type"),
                        # cancer 선택
                        dcc.Dropdown(
                            id="select-cancer-type",
                            options=[
                                {"label": i, "value": i} for i in list(content.keys())
                            ],
                            searchable=False,
                            value="Stomach",
                        ),
                        html.Br(),
                        html.H3("  SLC family"),
                        dcc.Dropdown(
                            id="slc-family-dropdown",
                            multi=True,
                            # value=[],
                        ),
                        # SLC 선택
                        html.Div(id="select-slc-family"),
                    ],
                    style={
                        "columnCount": 1,
                        "width": "15%",
                        # "display": "inline-block",  # inline-block: 해당 블록에 들어옴
                        # "float": "left",  # 위치 왼쪽
                        "textAlign": "center",
                    },
                ),
                html.Div(
                    [
                        html.H3(" Correlation"),
                        html.Label(" 선택한 암에 대한 SLC Biomarker 간의 Correlation"),
                        dcc.Graph(id="draw-cor", style={"": ""}),
                    ],
                    style={
                        "width": "40%",
                        # "display": "inline-block",
                        # "float": "left",
                        "textAlign": "center",
                    },  # block2 내 왼쪽
                ),
            ],
            style={
                "display": "flex",
                "flex-direction": "row",
                "justify-content": "center",
                # "align-items": "center",
            },
        ),
        html.Br(),
        html.Br(),
        html.Div(
            [
                html.Div(
                    [
                        html.Br(),
                        html.H3("Survival Anaylsis"),
                        # html.Label("Survival Anaylsis histogram"),
                    ],
                    style={"textAlign": "center"},
                ),
                html.Div(
                    [
                        html.H4("OS"),
                        dcc.Graph(id="draw-os-time", style={"width": "25%"}),
                    ],
                    style={
                        # "width": "25%",
                        "display": "inline-block",
                        "float": "left",
                        "textAlign": "center",
                    },  # block2 내 왼쪽
                ),
                html.Div(
                    [
                        html.H4("DSS"),
                        dcc.Graph(id="draw-dss-time", style={"width": "25%"}),
                    ],
                    style={
                        # "width": "25%",
                        "display": "inline-block",
                        "float": "left",
                        "textAlign": "center",
                    },  # block2 내 왼쪽
                ),
                html.Div(
                    [
                        html.H4("DFI"),
                        dcc.Graph(id="draw-dfi-time", style={"width": "25%"}),
                    ],
                    style={
                        # "width": "25%",
                        "display": "inline-block",
                        "float": "left",
                        "textAlign": "center",
                    },  # block2 내 왼쪽
                ),
                html.Div(
                    [
                        html.H4("PFI"),
                        dcc.Graph(id="draw-pfi-time", style={"width": "25%"}),
                    ],
                    style={
                        # "width": "25%",
                        "display": "inline-block",
                        "float": "left",
                        "textAlign": "center",
                    },  # block2 내 왼쪽
                ),
            ],
            style={
                "width": "100%",
                "display": "inline-block",
            },  # block2 내 왼쪽
        ),
    ]
)

# pca & tsne에 쓰이는 데이터 전처리


def preprocessing_PCA_TSNE(list_of_slc, list_of_cancer):
    # filtering
    df_1 = df[df["label_GTEx_100"] != 100]
    df_1 = pd.concat([df_1[["cancer", "label"]], df_1[list_of_slc]], axis=1)
    df_1 = df_1[df_1["cancer"].isin(list_of_cancer)]
    df_1.reset_index(drop=True, inplace=True)

    X = df_1.iloc[:, 1:]

    return df_1, X


####### PCA #######
@app.callback(
    Output("draw-PCA", "figure"),  # figure로 반환
    Input("select-slc-family", "value"),  # 값으로 입력
    Input("select-cancer-type", "value"),
)
# pca 그리기
def draw_PCA(list_of_slc, list_of_cancer):
    df_PCA, X = preprocessing_PCA_TSNE(
        df.columns[:-5].tolist(), df["cancer"].unique().tolist()
    )
    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    components = pd.DataFrame(components, columns=["f1", "f2"])

    # draw
    fig = px.scatter(
        components,
        x="f1",
        y="f2",
        color=df_PCA["cancer"],  # str
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    fig.update_traces(mode="markers", marker=dict(size=5))

    return fig


####### TSNE #######
@app.callback(
    Output("draw-TSNE", "figure"),
    Input("select-slc-family", "value"),  # 값으로 입력
    Input("select-cancer-type", "value"),
)

# tsne 그리기
def draw_TSNE(list_of_slc, list_of_cancer):
    df_TSNE, X = preprocessing_PCA_TSNE(
        df.columns[:-5].tolist(), df["cancer"].unique().tolist()
    )
    # TSNE
    tsne = TSNE(n_components=2, random_state=0)
    components = tsne.fit_transform(X)
    components = pd.DataFrame(components, columns=["f1", "f2"])

    # draw
    fig = px.scatter(
        components,
        x="f1",
        y="f2",
        color=df_TSNE["cancer"],  # str
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    fig.update_traces(mode="markers", marker=dict(size=5))

    return fig


@app.callback(
    Output("slc-family-dropdown", "options"),
    Output("slc-family-dropdown", "value"),
    Input("select-slc-family", "value"),  # 값으로 입력
    Input("select-cancer-type", "value"),
)
def make_list(list_of_slc, list_of_cancer):
    opt = [{"label": i, "value": i} for i in content[list_of_cancer]]
    val = content[list_of_cancer]
    return opt, val


@app.callback(
    Output("draw-cor", "figure"),
    Input("slc-family-dropdown", "value"),
    Input("select-cancer-type", "value"),
)
def corr(list_of_slc, list_of_cancer):
    df_cor = df[list_of_slc]
    # print(df_cor)
    df_corr = df_cor.corr(method="pearson")
    # print(df_corr)

    columns = list(df_corr.columns.values)
    rows = list(df_corr.index)

    clustergram = dashbio.Clustergram(
        data=df_corr.loc[rows].values,
        row_labels=rows,
        column_labels=columns,
        color_threshold={"row": 250, "col": 700},
        height=600,
        width=750,
        color_map=[
            [0.0, "#636EFA"],
            [0.25, "#AB63FA"],
            [0.5, "#FFFFFF"],
            [0.75, "#E763FA"],
            [1.0, "#EF553B"],
        ],
    )
    # dcc.Graph(figure=clustergram)

    return clustergram


@app.callback(
    Output("draw-os-time", "figure"),
    Input("slc-family-dropdown", "value"),
    Input("select-cancer-type", "value"),
)
def draw_os_time(list_of_slc, list_of_cancer):
    # list_of_cancer
    # df2
    # print(list_of_cancer)
    dc = df2[df2["cancer"] == list_of_cancer]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            name="All of cancer",
            x=df2["OS.time"],
            opacity=0.8,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Histogram(
            name=list_of_cancer,
            x=dc["OS.time"],
            opacity=0.8,
            showlegend=False,
        )
    )

    # The two histograms are drawn on top of another
    fig.update_layout(
        barmode="overlay",
        width=400,
        height=400,
        margin=dict(t=0),
    )

    # fig.show()

    return fig


@app.callback(
    Output("draw-dss-time", "figure"),
    Input("slc-family-dropdown", "value"),
    Input("select-cancer-type", "value"),
)
def draw_dss_time(list_of_slc, list_of_cancer):
    # list_of_cancer
    # df2
    # print(list_of_cancer)
    dc = df2[df2["cancer"] == list_of_cancer]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            name="All of cancer",
            x=df2["DSS.time"],
            opacity=0.8,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Histogram(
            name=list_of_cancer,
            x=dc["DSS.time"],
            opacity=0.8,
            showlegend=False,
        )
    )

    # The two histograms are drawn on top of another
    fig.update_layout(
        barmode="overlay",
        width=400,
        height=400,
        margin=dict(t=0),
    )
    # fig.show()

    return fig


@app.callback(
    Output("draw-dfi-time", "figure"),
    Input("slc-family-dropdown", "value"),
    Input("select-cancer-type", "value"),
)
def draw_dfi_time(list_of_slc, list_of_cancer):
    # list_of_cancer
    # df2
    # print(list_of_cancer)
    dc = df2[df2["cancer"] == list_of_cancer]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            name="All of cancer",
            x=df2["DFI.time"],
            opacity=0.8,
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Histogram(
            name=list_of_cancer,
            x=dc["DFI.time"],
            opacity=0.8,
            showlegend=False,
        )
    )

    # The two histograms are drawn on top of another
    fig.update_layout(
        barmode="overlay",
        width=400,
        height=400,
        margin=dict(t=0),
    )
    # fig.show()

    return fig


@app.callback(
    Output("draw-pfi-time", "figure"),
    Input("slc-family-dropdown", "value"),
    Input("select-cancer-type", "value"),
)
def draw_pfi_time(list_of_slc, list_of_cancer):
    # list_of_cancer
    # df2
    # print(list_of_cancer)
    dc = df2[df2["cancer"] == list_of_cancer]

    fig = go.Figure()
    fig.add_trace(go.Histogram(name="All of cancer", x=df2["PFI.time"], opacity=0.8))
    fig.add_trace(go.Histogram(name=list_of_cancer, x=dc["PFI.time"], opacity=0.8))

    # The two histograms are drawn on top of another
    fig.update_layout(
        barmode="overlay",
        width=400,
        height=400,
        margin=dict(t=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.48),
    )
    # fig.show()

    return fig
