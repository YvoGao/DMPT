import os
import random
import numpy as np
import datetime
import pytz
from tqdm import tqdm
from pprint import pprint

import torch
import torch.nn as nn

import models
from pengi import pengi


from utils import trainer
from utils.utils import print_total_time, get_args, get_dataloaders, get_model, setup_logging, get_scores, print_scores, save_scores, load_model

# to solve  the issue of : the current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):

    print(f"\n\n{'Model:':<10}{args.model_name.upper()}")
    print(f"{'Dataset:':<10}{args.dataset_root.split('/')[-1]}")
    print(f"{'Seed:':<10}{args.seed}\n\n")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    args.process_audio_fn = pengi.preprocess_audio

    # to ensure reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
    train_dataloader, test_dataloader = get_dataloaders(args)
    args.classnames = train_dataloader.dataset.classnames
    assert train_dataloader.dataset.classnames == test_dataloader.dataset.classnames, "Classnames in train and test datasets are different."

    model = get_model(args, pengi, models)
    model.to(device)
    # import pdb; pdb.set_trace()
    criterion = nn.CrossEntropyLoss()
    
    print("\nArguments:\n")
    for arg in vars(args): print(f"{arg:<25}: {getattr(args, arg)}")
    print("\n\n")


    if args.eval_only or args.model_name == "zeroshot":
        if args.model_name != "zeroshot": 
            load_model(args, model)
        test_loss, actual_labels, predicted_labels = trainer.run_evaluation(model, test_dataloader, criterion, device)
        accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
        print(f"\n\n-------------------------------\nTest Evaluation\n-------------------------------\n")
        print_scores(accuracy, f1_score, precision, recall, test_loss)
        if args.do_logging:
            print("Saving Results ...") 
            save_scores(args.seed, -1, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
            print("Results Saved\n\n")
    else:
        optimizer = torch.optim.Adam(model.prompt_learner.parameters(), lr=args.lr)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.model_name == 'pnalm':
            args.n_epochs = 2
        trainer.run_training(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=args.n_epochs, args=args)



if __name__ == "__main__":

    args = get_args()
    

    print("\n\n##############################################")
    print("PALM: Prompt Learning in Audio Language Models")
    print("##############################################\n\n")
    date_now = datetime.datetime.now(pytz.timezone('Asia/Dubai'))
    print(f'Time & Date = {date_now.strftime("%I:%M %p")} , {date_now.strftime("%d_%b_%Y")}  GST\n')
    # for w in range(10):
    #     args.weight = w / 10.0
    for shot in [16]:
        args.num_shots = shot
        log_file = setup_logging(args)
        print(f"\n\n********** {args.num_shots}-SHOT LEARNING **********\n\n")
        main(args)

