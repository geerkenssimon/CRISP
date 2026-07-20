# build-in modules
import numpy as np

# third party modules
import matplotlib.patches as patches
from matplotlib import widgets, backend_bases, path
from matplotlib.axes._axes import Axes

class Rectangle(widgets.RectangleSelector):
    _props = dict(fill=False, edgecolor='red', snap=True)
    def __init__(self, ax:Axes, callback, useblit:bool=True):
        super().__init__(ax, self._callback, 
                         useblit=useblit,
                         props=dict(fill=False, edgecolor='red', snap=True), 
                         spancoords='data',
                         button=[1], grab_range=3,
                         interactive=False,
                         use_data_coordinates=True,)
        self.callback = callback
        self.ax = ax
        self.current_rect = None
        
    def press(self, event:backend_bases.MouseEvent):
        """Button press handler and validator."""
        if self.current_rect is not None:
            self.current_rect.remove()
        if not self.ignore(event):
            event = self._clean_event(event)
            int_event = event
            xpre = event.xdata - np.around(int_event.xdata, decimals=0)
            ypre = event.ydata - np.around(int_event.ydata, decimals=0)
            int_event.xdata = np.around(int_event.xdata, decimals=0) + .5*np.sign(xpre)
            int_event.ydata = np.around(int_event.ydata, decimals=0) + .5*np.sign(ypre)
            self._eventpress = int_event
            self._prev_event = int_event
            key = event.key or ''
            key = key.replace('ctrl', 'control')
            # move state is locked in on a button press
            if key == self._state_modifier_keys['move']:
                self._state.add('move')
            self._press(event)
            return True
        return False

    def release(self, event:backend_bases.MouseEvent):
        """Button release event handler and validator."""
        if not self.ignore(event) and self._eventpress:
            event = self._clean_event(event)
            int_event = event
            xpre = event.xdata - np.around(int_event.xdata, decimals=0)
            ypre = event.ydata - np.around(int_event.ydata, decimals=0)
            int_event.xdata = np.around(int_event.xdata, decimals=0) + .5*np.sign(xpre)
            int_event.ydata = np.around(int_event.ydata, decimals=0) + .5*np.sign(ypre)
            self._eventrelease = int_event
            self._release(event)
            self._eventpress = None
            self._eventrelease = None
            self._state.discard('move')
            return True
        return False

    def onmove(self, event:backend_bases.MouseEvent):
        """Cursor move event handler and validator."""
        if not self.ignore(event) and self._eventpress:
            event = self._clean_event(event)
            int_event = event
            xpre = event.xdata - np.around(int_event.xdata, decimals=0)
            ypre = event.ydata - np.around(int_event.ydata, decimals=0)
            int_event.xdata = np.around(int_event.xdata, decimals=0) + .5*np.sign(xpre)
            int_event.ydata = np.around(int_event.ydata, decimals=0) + .5*np.sign(ypre)
            self._onmove(int_event)
            return True
        return False
    
    def _callback(self, eclick:backend_bases.MouseEvent, 
                             erelease:backend_bases.MouseEvent):
        """callback function for calculating bottom-left and tr corners of the drawn rectangle.
        Called when a rectangle has been drawn to the histogram.

        Parameters
        ----------
        eclick : backend_bases.MouseEvent
            object of the MouseEvent when clicking 
        erelease : backend_bases.MouseEvent
            object of the MouseEvent when releasing 
        """
        # receiving the coordinates of click and release from drawing the
        # rectangle on the plot.
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Compute bottom-left and width/height
        bl_x, bl_y = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        
        # Create a persistent rectangle
        self.current_rect = patches.Rectangle((bl_x, bl_y), width, height, 
                                **self._props)
        self.ax.add_patch(self.current_rect)  # Add rectangle to the axis

        # computing bottom-left (bl) and top-right (tr) corners of the rectangle.
        bl = (int(abs(min(np.ceil(x1),np.ceil(x2)))), int(abs(min(np.ceil(y1),np.ceil(y2)))))
        tr = (int(abs(max(np.floor(x1),np.floor(x2)))), int(abs(max(np.floor(y1),np.floor(y2)))))
        
        self.callback(bl, tr)
    

class Polygon(widgets.PolygonSelector):
    def __init__(self, ax:Axes, callback, useblit:bool=True):
        super().__init__(ax, self._callback, 
                         props={'color': 'r'}, 
                         handle_props={'markersize': 2}, 
                         useblit=useblit)
        self.callback = callback

        x1, x2 = ax.axes.get_xlim()
        y1, y2 = ax.axes.get_ylim()

        xx, yy = np.meshgrid(np.ceil(np.arange(x1, x2)), np.ceil(np.arange(y1,y2, np.max((1, (y2-y1)//200)))))

        self.xys = np.dstack([ xx, yy ]).reshape(-1,2)
        self.Npts = len(self.xys)
    
    def _callback(self, verts):
        p = path.Path(verts)
        ind = np.nonzero(p.contains_points(self.xys))[0]
        a = self.xys[ind]
        ks = np.unique(a[:,0])
        vs = [(int(np.min(a[:,1][a[:,0]==k])), 
              int(np.max(a[:,1][a[:,0]==k])+1)) for k in ks]
        self.callback(ks, vs)
