
class CFG:

    ## Melspectrogram ##
    sample_rate = 32000
    n_fft = 2048
    win_length = None
    hop_length = 1024
    n_mels = 128
    segment_train = sample_rate*5
    segment_test = sample_rate*5

    ## Train ##
    num_epochs = 1
    lr = 0.001
    batch_size = 16

    ## Input Data ##
    root_path = "../input/birdclef-2022/"
    input_path = root_path + '/train_audio/'
    out_train_path = "./train/"
    out_val_path = "./val/"
    test_audio_path = '../input/birdclef-2022/test_soundscapes/'
    model_out_path = './models/'


