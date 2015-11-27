import download_pretrained_model
import build_reconstructor
import reconstruct
import compute_saliency_map


def run():
    if True:
        download_pretrained_model.run()

    if True:
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
