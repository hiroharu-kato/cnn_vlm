"""
Run all scripts
"""

import build_alexnet
import build_vggnet
import prepare_images
import compute_mean_std
import train_pca
import train_vlm
import build_reconstructor
import reconstruct
import compute_saliency_map


def run():
    if False:
        print 'build theano models'
        build_alexnet.run()
        build_vggnet.run()

    if False:
        print 'prepare images'
        prepare_images.run()

    if False:
        print 'compute mean and std of feature maps'
        compute_mean_std.run(
            model_name='alexnet',
            layer_names=[
                'data',
                'conv1',
                'conv2',
                'conv3',
                'conv4',
                'conv5',
                'fc6', 'fc7', 'fc8',
            ],
            image_size=(227, 227),
        )
        compute_mean_std.run(
            model_name='vggnet',
            layer_names=[
                'data',
                'conv1_1', 'conv1_2',
                'conv2_1', 'conv2_2',
                'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                'fc6', 'fc7', 'fc8'
            ],
            image_size=(224, 224),
        )

    if False:
        print 'PCA of feature maps'
        train_pca.run(
            model_name='alexnet',
            layer_names=[
                'data',
                'conv1',
                'conv2',
                'conv3',
                'conv4',
                'conv5'
            ],
            image_size=(227, 227),
        )
        train_pca.run(
            model_name='vggnet',
            layer_names=[
                'data',
                'conv1_1', 'conv1_2',
                'conv2_1', 'conv2_2',
                'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
            ],
            image_size=(224, 224),
        )

    if False:
        print 'Train language model'
        train_vlm.run(
            model_name='alexnet',
            layer_names=['data', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
            layer_sizes=[3, 96, 256, 384, 384, 256],
            image_size=(227, 227),
        )
        train_vlm.run(
            model_name='vggnet',
            layer_names=[
                'data',
                'conv1_1',
                'conv2_1',
                'conv3_1',
                'conv4_1',
                'conv5_1',
            ],
            layer_sizes=[3, 64, 128, 256, 512, 512],
            batch_size=1,
            lr=2e+1,
            image_size=(224, 224),
        )
        train_vlm.run(
            model_name='vggnet',
            layer_names=[
                'data',
                'conv1_2',
                'conv2_2',
                'conv3_4',
                'conv4_4',
                'conv5_4',
            ],
            layer_sizes=[3, 64, 128, 256, 512, 512],
            batch_size=1,
            lr=2e+1,
            image_size=(224, 224),
        )

    if False:
        print 'Bulid reconstructor'
        build_reconstructor.run(
            model_name='alexnet',
            layer_names=[
                'data',
                'conv1',
                'conv2',
                'conv3',
                'conv4',
                'conv5'
            ],
            layer_target=[
                'conv1',
                'conv2',
                'conv3',
                'conv4',
                'conv5',
            ],
            image_size=(227, 227),
            target_layer='fc8',
        )

    if True:
        print 'Test models'
        reconstruct.run('./testdata/024_227.png', 'reconstructed.png')
        compute_saliency_map.run('./testdata/024.jpg', 'saliency_map.png')

if __name__ == '__main__':
    run()
