import altair as alt


def rapids_theme():
    font = "Open Sans"
    text_color = "#666666"
    main_palette = ["#7400ff", "#36c9dd", "#d216d2", "#ffb500"]
    secondary_palette = ["#bababc", "#666666", "#8824ff", "#9942ff", "#a785e7"]

    return {
        "config": {
            "axis": {
                "labelFontSize:": 20,
                "labelColor": text_color,
                "titleFontSize": 20,
                "titleColor": text_color,
            },
            "axisY": {
                "font": font,
                "labelFontSize": 20,
                "labelColor": text_color,
                "titleFontSize": 20,
                "titleColor": text_color,
            },
            "axisX": {
                "font": font,
                "labelFontSize": 20,
                "labelColor": text_color,
                "titleFontSize": 20,
                "titleColor": text_color,
            },
            "header": {
                "font": font,
                "labelFontSize": 20,
                "labelColor": text_color,
                "titleFontSize": 20,
                "titleColor": text_color,
            },
            "legend": {
                "font": font,
                "labelFontSize": 18,
                "labelColor": text_color,
                "titleFontSize": 18,
                "titleColor": text_color,
                "strokeColor": text_color,
                "padding": 10,
            },
            "range": {
                "category": main_palette,
                "diverging": secondary_palette,
            },
        }
    }


alt.themes.register("RAPIDS", rapids_theme)
