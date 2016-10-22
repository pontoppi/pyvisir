import matplotlib.pylab as plt
from pyvisir import reduction,visir_run


path = '/astro/pontoppi/DATA/VISIR2.0/095.C-0203'
vrun = visir_run.visir_run(path)
#scilist = vrun.get_sciencelist(1368832)
scilist = vrun.get_sciencelist(1368835)
stdlist = vrun.get_caliblist(1368838)

Red = reduction.Reduction(scilist=scilist,stdlist=stdlist,
                          level1=True, level2=False, level3=False,save_flat=False)
