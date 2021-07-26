import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

from app import app

import pandas as pd
import pickle
import joblib

import plotly.express as px
import plotly.graph_objects as go

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow import keras


pd.set_option("display.float_format", "{:.5f}".format)


# 데이터 불러오기 (전역변수처럼)
data = pd.read_csv("./apps/TCGA_GTEX_SLC_103.csv")


# 레이아웃: html 코드
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
        # 새로운 자료 입력
        html.Div(
            [
                html.H2(
                    "SLC 발현량 csv 파일을 업로드해주세요",
                    style={"textAlign": "center"},
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style={
                        "width": "100%",
                        "height": "70px",
                        "lineHeight": "70px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        # "margin": "10px",
                    },
                    # Allow multiple files to be uploaded
                    # multiple=True,
                ),
            ],
            style={"width": "100%", "display": "block", "padding": "30px 0px"},
        ),
        # 중간
        html.Div(
            [
                # maching learning 버튼 느낌으로
                html.Div(
                    [
                        html.H3("Machine Learning"),
                        html.Label("19개 암종 및 정상을 예측한 결과입니다."),
                        html.Br(),
                        html.Br(),
                        html.Div(
                            [
                                html.H3("Linear SVM"),
                                html.Div(
                                    [
                                        html.Div(id="linear-svm"),
                                    ],
                                    style={"color": "white"},
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "textAlign": "center",
                                "background": "#87c4d5",
                                "padding": "10px 55px",
                                "margin": "10px",
                                "opacity": 0.8,
                            },
                        ),
                        html.Div(
                            [
                                html.H3("XGBoost"),
                                html.Div(
                                    [
                                        html.Div(id="xgboost"),
                                    ],
                                    style={"color": "white"},
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "textAlign": "center",
                                "background": "#87c4d5",
                                "padding": "10px 55px",
                                "margin": "10px",
                                "opacity": 0.8,
                            },
                        ),
                        html.Div(
                            [
                                html.H3("Random Forest"),
                                html.Div(
                                    [
                                        html.Div(id="random-forest"),
                                    ],
                                    style={"color": "white"},
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "textAlign": "center",
                                "background": "#87c4d5",
                                "padding": "10px 55px",
                                "margin": "10px",
                                "opacity": 0.8,
                            },
                        ),
                        html.Div(
                            [
                                html.H3("K-NN"),
                                html.Div(
                                    [
                                        html.Div(id="k-nn"),
                                    ],
                                    style={"color": "white"},
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "textAlign": "center",
                                "background": "#87c4d5",
                                "padding": "10px 55px",
                                "margin": "10px",
                                "opacity": 0.8,
                            },
                        ),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "float": "left",
                        "textAlign": "center",
                    },
                ),
                # CNN bar chart 출력
                html.Div(
                    [
                        html.H3("1D-CNN"),
                        html.Label("각 암종 및 정상으로 예측될 확률입니다."),
                        dcc.Graph(id="output-cnn"),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "float": "right",
                        "textAlign": "center",
                    },
                ),
            ],
            style={"width": "100%", "display": "inline-block", "textAlign": "center"},
        ),
        # 아래쪽
        html.Div(
            [
                # pca
                html.Div(
                    [
                        html.H3("PCA(Component=2)"),
                        html.Label("업로드한 SLC 발현량에 대한 PCA 결과는 square 모양으로 나타납니다."),
                        dcc.Graph(id="draw-PCA-input", style={"height": "65vh"}),
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
                        html.Label("업로드한 SLC 발현량에 대한 t-SNE 결과는 square 모양으로 나타납니다."),
                        dcc.Graph(id="draw-TSNE-input", style={"height": "65vh"}),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "float": "right",
                        "textAlign": "center",
                    },
                ),
            ],
            style={"width": "100%"},
        ),
    ],
)


# pca & tsne에 쓰이는 데이터 전처리
def preprocessing_PCA_TSNE(df):
    # filtering
    df = df[df["label_GTEx_100"] != 100]
    df.drop(
        columns=["sample", "TCGA_GTEX_main_category", "label_GTEx_100", "label"],
        inplace=True,
    )
    df.reset_index(drop=True, inplace=True)

    X = df.iloc[:, :-2]

    return df, X


# 입력된 데이터를 df 형식으로 받아오는 함수


def make_df(content, filename):
    content_type, content_string = content.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        return df
    except Exception as e:
        print(e)
        return


# 머신러닝에 사용되는 input data 생성하는 함수


def preprocessing_for_machine(df):
    df_m = df.drop(columns=["sample", "TCGA_GTEX_main_category", "cancer", "label"])
    df_m.rename(columns={"label_GTEx_100": "label"}, inplace=True)
    # split X, y
    X = df_m.iloc[:, :-1]
    y = df_m.iloc[:, -1:]

    return X, y


# CNN에 사용되는 input data 생성하는 함수


def preprocessing_for_cnn(df):
    df_m = df.drop(columns=["sample", "TCGA_GTEX_main_category", "cancer", "label"])
    df_m.rename(columns={"label_GTEx_100": "label"}, inplace=True)
    # label 100 -> 19
    df_m.loc[df_m["label"] == 100, "label"] = 19
    # split X, y
    X = df_m.drop(columns=["label"]).to_numpy()

    X = X.reshape(X.shape[0], 1, 103, 1)  # 12829, 1, 103, 1
    X = X.astype("float32")

    return X


@app.callback(
    Output("linear-svm", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
# 받아온 데이터를 가지고 머신러닝 진행
def machine_learning(content, filename):
    if content is not None:
        df = make_df(content, filename)  # 전처리 함수 call
        X, y = preprocessing_for_machine(df)

        # load model
        svm_model = joblib.load("./models/svm_model.pkl", "r")

        # predict
        svm_pred = df.loc[
            df["label_GTEx_100"] == svm_model.predict(X)[0], "cancer"
        ].unique()[0]

        return html.H1(svm_pred)


@app.callback(
    Output("xgboost", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
# 받아온 데이터를 가지고 머신러닝 진행
def machine_learning(content, filename):
    if content is not None:
        df = make_df(content, filename)  # 전처리 함수 call
        X, y = preprocessing_for_machine(df)

        # load model
        xgb_model = joblib.load("./models/xgb_model.pkl", "r")

        # predict
        xgb_pred = df.loc[
            df["label_GTEx_100"] == xgb_model.predict(X)[0], "cancer"
        ].unique()[0]

        return html.H1(xgb_pred)


@app.callback(
    Output("random-forest", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
# 받아온 데이터를 가지고 머신러닝 진행
def machine_learning(content, filename):
    if content is not None:
        df = make_df(content, filename)  # 전처리 함수 call
        X, y = preprocessing_for_machine(df)

        # load model
        rf_model = joblib.load("./models/rf_model.pkl", "r")

        # predict
        rf_pred = df.loc[
            df["label_GTEx_100"] == rf_model.predict(X)[0], "cancer"
        ].unique()[0]

        return html.H1(rf_pred)


@app.callback(
    Output("k-nn", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
# 받아온 데이터를 가지고 머신러닝 진행
def machine_learning(content, filename):
    if content is not None:
        df = make_df(content, filename)  # 전처리 함수 call
        X, y = preprocessing_for_machine(df)

        # load model
        knn_model = joblib.load("./models/knn_model.pkl", "r")

        # predict
        knn_pred = df.loc[
            df["label_GTEx_100"] == knn_model.predict(X)[0], "cancer"
        ].unique()[0]

        return html.H1(knn_pred)


# cnn 부분인데 아직 수정 다 못함
@app.callback(
    Output("output-cnn", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def cnn(content, filename):
    if content is not None:
        df = make_df(content, filename)
        X = preprocessing_for_cnn(df)
        # load model
        cnn_model = tf.keras.models.load_model("./models/cnn_model.h5")
        # predict
        y_pred = cnn_model.predict(X)[0]
        df_pred = pd.DataFrame(columns=["Cancer", "Probability"], index=range(0))
        df_pred["Cancer"] = [
            "Adrenal",
            "Bladder",
            "Breast",
            "Cervix",
            "Colon",
            "Blood",
            "Esophagus",
            "Brain",
            "Kidney",
            "Liver",
            "Lung",
            "Ovary",
            "Pancreas",
            "Prostate",
            "Skin",
            "Stomach",
            "Testis",
            "Thyroid",
            "Uterus",
            "Normal",
        ]
        df_pred["Probability"] = y_pred
        df_pred.sort_values(by="Probability", ascending=True, inplace=True)
        df_pred["color"] = "blue"
        df_pred.loc[df_pred["Probability"] >= 0.5, "color"] = "red"

        fig = px.bar(
            df_pred,
            y="Cancer",
            x="Probability",
            color="color",
            opacity=0.8,
            text=df_pred["Probability"],
        )
        fig.update_traces(
            texttemplate="%{text:.2f}", textposition="outside", showlegend=False
        )
        fig.update_layout(uniformtext_minsize=8, margin=dict(r=10))

        return fig

    else:
        fig = px.bar()
        return fig


##########################################################################
@app.callback(
    Output("draw-PCA-input", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
# pca 그리기
def draw_PCA(content, filename):
    if content is not None:
        df = make_df(content, filename)  # 전처리 함수 call
        # input data label 변경
        df["cancer"] = "Input"
        df["input_yn"] = "y"
        data["input_yn"] = "n"
        df_1 = pd.concat([data, df], axis=0)
        df_PCA, X = preprocessing_PCA_TSNE(df_1)
    else:
        data["input_yn"] = "n"
        df_PCA, X = preprocessing_PCA_TSNE(data)

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
        symbol=df_PCA["input_yn"],
        symbol_sequence=["x", "square"],
    )

    fig.update_traces(mode="markers", marker=dict(size=5))

    return fig


@app.callback(
    Output("draw-TSNE-input", "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
# tsne 그리기
def draw_TSNE(content, filename):
    if content is not None:
        df = make_df(content, filename)  # 전처리 함수 call
        # input data label 변경
        df["cancer"] = "Input"
        df["input_yn"] = "y"
        data["input_yn"] = "n"
        df_1 = pd.concat([data, df], axis=0)
        df_TSNE, X = preprocessing_PCA_TSNE(df_1)
    else:
        data["input_yn"] = "n"
        df_TSNE, X = preprocessing_PCA_TSNE(data)

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
        symbol=df_TSNE["input_yn"],
        symbol_sequence=["x", "square"],
    )

    fig.update_traces(mode="markers", marker=dict(size=5))

    return fig
