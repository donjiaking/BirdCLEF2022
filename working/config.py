class CFG:
    # Pre-process #
    noise_p = 0.5
    volume_p = 0.2
    normalize_p = 1
    noise_level = (0, 0.5)
    db_limit = 10
    mean = 0
    sigma = 0.1
    # mixup_p = 0.5
    # alpha = 0.5

    # Melspectrogram #
    sample_rate = 32000
    n_fft = 2048
    win_length = None
    hop_length = 1024
    n_mels = 128
    segment_train = sample_rate * 30  # 5
    segment_test = sample_rate * 5

    # Train #
    num_epochs = 15
    lr = 1e-5
    batch_size = 3  # 16
    print_feq = 100

    n_classes = 152
    backbone = 'resnet50'  # 'resnext101_32x8d' 'efficientnet_b0' - following timm's name
    pretrained = True
    mix_beta = 1

    # Input Data #
    root_path = "../input/birdclef-2022/"
    input_path = root_path + 'train_audio/'
    out_train_path = "./train/"
    out_val_path = "./val/"
    test_audio_path = '../input/birdclef-2022/test_soundscapes/'
    model_out_path = './models/'

    # Binary Threshold #
    binary_th = 0.3

    
