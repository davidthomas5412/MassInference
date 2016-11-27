from os import mkdir, path, remove
from traceback import print_exc
from requests import get

MODULE_DIR = path.dirname(path.realpath(__file__))
DATA_DIR = path.join(path.dirname(MODULE_DIR), 'data')

GAMMA_1_FILE = path.join(DATA_DIR, 'GGL_los_8_0_0_N_4096_ang_4_rays_to_plane_37_f.gamma_1')
GAMMA_2_FILE = path.join(DATA_DIR, 'GGL_los_8_0_0_N_4096_ang_4_rays_to_plane_37_f.gamma_2')
KAPPA_FILE = path.join(DATA_DIR, 'GGL_los_8_0_0_N_4096_ang_4_rays_to_plane_37_f.kappa')
GUO_FILE = path.join(DATA_DIR, 'GGL_los_8_0_0_0_0_N_4096_ang_4_Guo_galaxies_on_plane_27_to_63.images.txt')

DATA_FILES = [GAMMA_1_FILE, GAMMA_2_FILE, KAPPA_FILE, GUO_FILE]

def fetch():
    """
    Downloads the data for both the demo notebooks and example configuration.
    """
    data_url = 'http://www.slac.stanford.edu/~pjm/hilbert'

    for d in [DATA_DIR]:
        if not path.exists(d):
            mkdir(d)

    for demo_file in DATA_FILES:
        if not path.exists(demo_file):
            url = '{}/{}'.format(data_url, path.basename(demo_file))
            download(url, demo_file)

def download(url, output):
    """
    Downloads the data for both the demo notebooks and example configuration.

    Note:
        Does not provide progress feedback so be careful using this for HUGE downloads or SLOW network.

    Args:
    url (str): url to download from.
    output (str): path to write to.
    """
    print "Starting download \n" \
               "\t{}\n" \
               "\t >>> {}".format(url, output)
    try:
        r = get(url, stream=True)
        with open(output, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print "Done"
    except Exception:
        try:
            remove(output)
        except OSError:
            pass
        print_exc()
        msg = "Error downloading from '{}'. Please issue this on " \
              "https://github.com/davidthomas5412/MassInference/issues".format(url)
        raise Exception(msg)
