import getpass
import socket

def get_hci_home_path():
    username = getpass.getuser()
    hostname = socket.gethostname()
    if hostname == 'trendytukan' and username == 'abailoni':
        return '/net/hcihome/storage/abailoni/'
    elif hostname == 'trendytukan' and username == 'abailoni_local':
        # return '/home/abailoni_local/hci_home/'
        return '/home/abailoni_local/ialgpu1_local_home/'
    elif hostname == 'ialgpu01' or hostname == 'birdperson' or hostname == 'sirherny':
        return '/home/abailoni/local_copy_home/'
        # return '/home/abailoni/hci_home/'
    elif hostname == 'quadxeon5':
        return '/srv/scratch/abailoni'
    else:
        return '/net/hcihome/storage/abailoni/local_home/'

def get_trendytukan_drive_path():
    username = getpass.getuser()
    hostname = socket.gethostname()
    # print(username, hostname)
    if hostname == 'trendytukan' and username == 'abailoni':
        return '/mnt/localdata0/abailoni/'
    elif hostname == 'trendytukan' and username == 'abailoni_local':
        return '/home/abailoni_local/trendyTukan_localdata0/'
    elif hostname == 'ialgpu01' or hostname == 'birdperson' or hostname == 'sirherny':
        return '/home/abailoni/trendyTukan_drive/'
    elif hostname == 'quadxeon5':
        return '/srv/scratch/abailoni'
    else:
        return '/net/hcihome/storage/abailoni/trendyTukan_drive/'
