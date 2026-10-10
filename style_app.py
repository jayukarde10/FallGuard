import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace plotly backgrounds
code = code.replace("paper_bgcolor='rgba(15, 23, 42, 0.6)'", "paper_bgcolor='#FFFFFF'")
code = code.replace("plot_bgcolor='rgba(15, 23, 42, 0.6)'", "plot_bgcolor='#FFFFFF'")

# Replace plotly text colors
code = code.replace('color="#e2e8f0"', 'color="#191919"')

# Replace grid colors
code = code.replace("gridcolor='rgba(255, 255, 255, 0.08)'", "gridcolor='#DDD6CA'")
code = code.replace("zerolinecolor='rgba(255, 255, 255, 0.1)'", "zerolinecolor='#DDD6CA'")

# Replace timeline trace color (was blue #38bdf8, now terracotta #D9795E)
code = code.replace("color='#38bdf8'", "color='#D9795E'")
code = code.replace("fillcolor='rgba(56, 189, 248, 0.12)'", "fillcolor='rgba(217, 121, 94, 0.12)'")

# Replace histogram marker color
code = code.replace("marker_color='#818cf8'", "marker_color='#D9795E'")

# Replace metric card text colors that were hardcoded
code = code.replace('color: #38bdf8;', 'color: #D9795E;')
code = code.replace('color: #a855f7;', 'color: #77736B;')

# Replace font family Plus Jakarta Sans with Inter
code = code.replace('"Plus Jakarta Sans"', '"Inter"')

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)
