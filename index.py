import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app1, app2, app3


app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")],
)


index_page = html.Div(
    [
        html.Img(
            src="/assets/index.png",
            style={"width": "100%"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.A(
                            html.Img(
                                src="/assets/index_page1.jpeg",
                                width="100%",
                                style={"display": "block"},
                            ),
                            href="/apps/app1",
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.A(
                            html.Img(
                                src="/assets/index_page2.jpeg",
                                width="100%",
                                style={"display": "block"},
                            ),
                            href="/apps/app2",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.A(
                            html.Img(
                                src="/assets/index_page3.jpeg",
                                width="100%",
                                style={"display": "block"},
                            ),
                            href="/apps/app3",
                        ),
                    ],
                ),
            ],
            style={
                "display": "flex",
                "flex-direction": "row",
                "justify-content": "center",
                "align-items": "center",
            },
        ),
        html.Img(
            src="/assets/info.png",
            width="100%",
            style={"display": "block"},
        ),
    ],
    style={
        "display": "flex",
        "flex-direction": "column",
        "justify-content": "center",
        "align-items": "center",
        "font-size": "10px",
    },
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return index_page
    elif pathname == "/apps/app1":
        return app1.layout
    elif pathname == "/apps/app2":
        return app2.layout
    elif pathname == "/apps/app3":
        return app3.layout


if __name__ == "__main__":
    app.run_server(debug=True)
