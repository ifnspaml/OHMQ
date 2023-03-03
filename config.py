import os
class args(object):
    batch_size = 128
    n_epochs = 1000
    seed = 42
    learning_rate = 1e-3
    weight_decay = 1e-5
    num_workers = 6
    log_interval = 10
    val_interval = 1
    bottleneck_channels = [8, 8]
    bottleneck_bits = [2, 2]
    encoder_channels = 16
    no_quant_symbols = 32
    num_train_samples = -1
    num_val_samples = -1
    lambda_rd = [1]
    crop_size = 256
    data_base_path = os.getenv('data_base_path')
    checkpoint = 'checkpoints/example.pth'
    image_path = 'data/inference_images_gray'
    use_mnist_models = True
    skip = False
    channel_fraction_inference = 1
