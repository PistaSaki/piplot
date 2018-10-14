import io
import base64
def fig2html_src(fig):
    file = io.BytesIO()
    fig.savefig(file, format = "png")
    file.seek(0)
    img_str = base64.b64encode(file.read()).decode().replace("\\n", "")
    return "data:image/png;base64,{}".format(img_str)

def fig2html(fig):
    return '<img src="{}" alt="whatever">'.format(fig2html_src(fig))

#import matplotlib.pyplot as pl
#import numpy as np
#from IPython.core.display import HTML
#from IPython.display import display
#
#fig, ax = pl.subplots()
#xxx = np.linspace(-1, 1, 100)
#ax.plot(xxx, xxx**2)
#
#template = """
#<!DOCTYPE html>
#<html>
#<body>
#<p>This is a paragraph.</p>
#{}
#</body>
#</html>
#"""
#html = template.format(fig2html(fig))
#display(HTML(html))
