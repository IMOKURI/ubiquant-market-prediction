import numpy as np


try:
    import bottleneck

    nanmean = bottleneck.nanmean
    nanmin = bottleneck.nanmin
    nanmax = bottleneck.nanmax
    nanmedian = bottleneck.nanmedian
    median = bottleneck.median
    allnan = bottleneck.allnan
    nanstd = bottleneck.nanstd

except:
    nanmean = np.nanmean
    nanmin = np.nanmin
    nanmax = np.nanmax
    nanmedian = np.nanmedian
    median = np.median
    nanstd = np.nanstd

    def allnan(t):
        return np.all(np.isnan(t))
