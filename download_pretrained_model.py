import os
import subprocess


def run():
    directory = './theanomodel'
    url_alexnet = 'http://hiroharu-kato.com/download/cnn_vlm/alexnet_vlm_data_conv1_conv2_conv3_conv4_conv5.pkl'
    url_vggnet1 = 'http://hiroharu-kato.com/download/cnn_vlm/vggnet_vlm_data_conv1_1_conv2_1_conv3_1_conv4_1_conv5_1.pkl'
    url_vggnet2 = 'http://hiroharu-kato.com/download/cnn_vlm/vggnet_vlm_data_conv1_2_conv2_2_conv3_4_conv4_4_conv5_4.pkl'
    filename_alexnet = '%s/%s' % (directory, url_alexnet.split('/')[-1])
    filename_vggnet1 = '%s/%s' % (directory, url_vggnet1.split('/')[-1])
    filename_vggnet2 = '%s/%s' % (directory, url_vggnet2.split('/')[-1])
    if not os.path.exists(directory):
        os.mkdir(directory)

    if not os.path.exists(filename_alexnet):
        p = subprocess.Popen(('wget', url_alexnet, '-O', filename_alexnet))
        p.wait()

    if not os.path.exists(filename_vggnet1):
        p = subprocess.Popen(('wget', url_vggnet1, '-O', filename_vggnet1))
        p.wait()

    if not os.path.exists(filename_vggnet2):
        p = subprocess.Popen(('wget', url_vggnet2, '-O', filename_vggnet2))
        p.wait()


if __name__ == '__main__':
    run()
