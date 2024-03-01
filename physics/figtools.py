import pickle

def export_axis(ax, fp):
    exp_dat = {}
    #ax.get_children() # list all children
    #ax.fingobj(MATCH) # most general method to get object of certain type from ax
    # Export lines
    for i, line in enumerate(ax.get_lines()):
        x = line.get_xdata()
        y = line.get_ydata()
        exp_dat[f'line{i}'] = {'x':x, 'y':y}
    # Export collections e.g. scatter points
    for i, coll in enumerate(ax.collections):
        exp_dat[f'collection{i}'] = {'data': coll.get_offsets().data}
    # Export imshow objects
    for i, im in enumerate(ax.get_images()):
        extent = im.get_extent()
        cmap = im.get_cmap()
        data = im._A # _A contains data,_Ax, _Ay grid points
        # (don't need grid points if supply extent)
        # Can't see a public function get_data()
        # https://github.com/matplotlib/matplotlib/blob/v3.8.3/lib/matplotlib/image.py
        exp_dat[f'image{i}'] = {
                'extent': extent,
                'cmap': cmap.name, # just store name
                'data': data.data, # just store data, not masked array
                }
    with open(fp, 'wb') as fb:
        pickle.dump(exp_dat, fb)

# Example use for axes containing to matplotlib axis
EXPORT = False
if EXPORT:
    from figtools import export_axis
    fps = ['fig6.4b.pkl', 'fig6.4c.pkl'] # output data files
    for i, ax in enumerate(axes):
        export_axis(ax, fps[i])
