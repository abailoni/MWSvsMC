import matplotlib
matplotlib.use('Agg')

from .data_paths import get_hci_home_path
import sys
import os

sys.path += [
os.path.join(get_hci_home_path(), "python_libraries/nifty/python"),
os.path.join(get_hci_home_path(), "python_libraries/cremi_python"),
os.path.join(get_hci_home_path(), "python_libraries/affogato/python"),
os.path.join(get_hci_home_path(), "pyCharm_projects/inferno"),
os.path.join(get_hci_home_path(), "pyCharm_projects/constrained_mst"),
# os.path.join(get_hci_home_path(), "pyCharm_projects/neuro-skunkworks"),
os.path.join(get_hci_home_path(), "pyCharm_projects/segmfriends"),
# os.path.join(get_hci_home_path(), "pyCharm_projects/hc_segmentation"),
os.path.join(get_hci_home_path(), "pyCharm_projects/neurofire"),]

