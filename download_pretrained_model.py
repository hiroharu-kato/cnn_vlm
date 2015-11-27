import os
import subprocess


def run():
    url = 'http://hiroharu-kato.com/download/cnn_vlm/theanomodel.tar.gz'
    filename = './theanomodel.tar.gz'
    directory = './theanomodel'
    if not os.path.exists(filename):
        p = subprocess.Popen(('wget', url, '-O', filename))
        p.wait()

    if not os.path.exists(directory):
        p = subprocess.Popen(('tar', 'xvzf', filename))
        p.wait()

if __name__ == '__main__':
    run()
