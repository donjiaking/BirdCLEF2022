# import imp
# from logging.config import valid_ident
from train import train_loop
from utils import *
from sklearn.model_selection import train_test_split
from dataset import *


if __name__ == '__main__':
    # load data
    train_meta = pd.read_csv(CFG.meta_path)
    train_index, val_index = train_test_split(range(0, train_meta.shape[0]), train_size=0.8, test_size=0.2, random_state=42)

    # training
    oof_df, scores = train_loop(train_meta, train_index, val_index)
    
    # CV result
    # LOGGER.info(f"---------- CV ----------")
    print("========== CV Result ==========")
    get_result(oof_df)
    get_confusion_mat(oof_df)    
    
    # save result
    oof_df.to_csv('./oof_df.csv', index=False)
    plt.plot([i for i in range(CFG.epochs)], scores)
    plt.title('valid score')
    plt.show()

