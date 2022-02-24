try:
    import bottleneck as bn

    nanmean = bn.nanmean
    nanmax = bn.nanmax
    nanmin = bn.nanmin
    nanmedian = bn.nanmedian

except Exception:
    import numpy as np

    nanmean = np.nanmean
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanmedian = np.nanmedian
