import dash
import dash_html_components as html
from dash.dependencies import Input, Output

layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    html.A(
                        html.Img(src="/assets/nav_home.png"),
                        href="/",
                        style={"text-align": "center"},
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
        html.Img(src="/assets/eposter.jpeg", width="100%"),
    ],
    style={
        "display": "flex",
        "flex-direction": "column",
        "justify-content": "center",
        "align-items": "center",
    },
)
