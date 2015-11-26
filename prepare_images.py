import os
import glob
import shutil
import subprocess
import numpy as np
import caffe


def run():
    directory_temp = './temp'
    directory_train = './image/train'
    directory_valid = './image/valid'
    filename_train = '{{PATH_TO_DATASET}}/ILSVRC2012_img_train.tar'
    filename_valid = '{{PATH_TO_DATASET}}/ILSVRC2012_img_val.tar'
    image_height = 256
    image_width = 256
    url_mean = 'https://raw.githubusercontent.com/yosinski/convnet_transfer/master/results/transfer-ft1A1B_1_7/imagenet_mean.binaryproto'
    filename_caffe_mean = './caffemodel/imagenet_mean.binaryproto'
    filename_theano_mean = './theanomodel/imagenet_mean.npy'

    if not os.path.exists(filename_caffe_mean):
        print 'downloading mean file'
        p = subprocess.Popen(('wget', url_mean, '-O', filename_caffe_mean))
        p.wait()

    if not os.path.exists(filename_theano_mean):
        print 'converting mean file'
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.ParseFromString(open(filename_caffe_mean).read())
        mean = caffe.io.blobproto_to_array(blob)
        np.save(filename_theano_mean, mean)

    if False and not os.path.exists(directory_train):
        print 'unpacking ImageNet training set'

        # untar
        if not os.path.exists(directory_temp):
            os.mkdir(directory_temp)
        p = subprocess.Popen(('tar', 'xf', filename_train, '-C', directory_temp))
        p.wait()
        p = subprocess.Popen(
            (
                'find %s -name "*.tar" -exec tar xf {} -C %s \;' %
                (directory_temp, directory_temp)
            ),
            shell=True,
        )
        p.wait()

        # resize
        filenames = glob.glob('%s/*.JPEG' % directory_temp)
        for i, filename in enumerate(filenames):
            print 'resizing image: %d / %d' % (i, len(filenames))
            p = subprocess.Popen(
                (
                    'convert -scale %dx%d! %s %s' %
                    (image_width, image_height, filename, filename)
                ),
                shell=True,
            )
            p.wait()

        # move
        os.makedirs(directory_train)
        filenames = glob.glob('%s/*.JPEG' % directory_temp)
        for i, filename in enumerate(filenames):
            print 'moving image: %d / %d' % (i, len(filenames))
            p = subprocess.Popen(
                'mv %s %s/%s' % (filename, directory_train, filename.split('/')[-1]),
                shell=True,
            )
            p.wait()

        # delete temporary directory
        shutil.rmtree(directory_temp)

    if not os.path.exists(directory_valid):
        print 'unpacking ImageNet validation set'

        # untar
        if not os.path.exists(directory_temp):
            os.mkdir(directory_temp)
        p = subprocess.Popen(('tar', 'xf', filename_valid, '-C', directory_temp))
        p.wait()

        # resize
        p = subprocess.Popen(
            (
                'find %s -name "*.JPEG" -exec convert -scale %dx%d! {} {} \;' %
                (directory_temp, image_width, image_height)
            ),
            shell=True,
        )
        p.wait()

        # move
        os.makedirs(directory_valid)
        p = subprocess.Popen(
            'mv %s/*.JPEG %s/' % (directory_temp, directory_valid),
            shell=True,
        )
        p.wait()

        # delete temporary directory
        shutil.rmtree(directory_temp)

if __name__ == '__main__':
    run()
