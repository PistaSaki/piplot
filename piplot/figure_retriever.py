import os

import matplotlib.pyplot as pl

class FigureRetriever:
    def __init__(self, out_dir = None):
        self.out_dir = out_dir
        
    def __enter__(self):
        self._enter_fignums = pl.get_fignums()
        return self
        
    def __exit__(self, *args, **kwargs):
        self.fignums = [n for n in pl.get_fignums() if n not in self._enter_fignums]
        
        self.figures = figures = []
        for i in self.fignums:
            f = pl.figure(i)
            figures.append(f)
            
        if self.out_dir is not None:
            for i, fig in enumerate(self.figures):
                fig_title = fig._suptitle.get_text() if fig._suptitle is not None else ""
                fig.savefig(os.path.join(self.out_dir, f"fig_{i:03d}_{fig_title}.png"))
                
    def __iter__(self):
        return self.figures.__iter__()
        

    def close_figs(self):
        for fig in self.figures:
            pl.close(fig)
