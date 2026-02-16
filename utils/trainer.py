import os
import torch
import numpy as np
from tqdm import tqdm

from .utils import get_scores, print_scores, save_scores, timeit, save_model, get_save_model_path


def run_epoch(model, dataloader, optimizer, criterion, device, epoch, args=None):
    model.train()

    losses = []
    actual_labels = []
    predicted_labels = []

    for i, (audio, label) in enumerate(dataloader):

        audio = audio.to(device).squeeze(1)
        label = label.to(device)
 
        if args.model_name == "pealm" or args.model_name == "pnalm" or args.model_name == "claps":
            if epoch == 0:
                logits = model(audio, label, 'init')
            else:
                logits = model(audio, label, 'train')
        else:
            logits = model(audio)
        # import pdb; pdb.set_trace()
        loss = criterion(logits, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        actual_labels.extend(label.cpu().numpy())
        predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels




def init_epoch(model, dataloader, optimizer, criterion, device, epoch, args=None):
  

    for i, (audio, label) in enumerate(dataloader):

        audio = audio.to(device).squeeze(1)
        label = label.to(device)
        
        model.add_memory(label, audio)


@timeit
def run_evaluation(model, dataloader, criterion, device, args):
    model.eval()

    losses = []
    actual_labels = []
    predicted_labels = []
    
    print("\n\nEvaluating the model ...")
    with torch.no_grad():
        for i, (audio, label) in enumerate(dataloader):
            # import pdb; pdb.set_trace()
        # for i, (audio, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            print(f"Batch {i+1}/{len(dataloader)}")

            audio = audio.to(device).squeeze(1)
            label = label.to(device)
            
            if args.model_name == "pealm" or args.model_name == "pnalm" or args.model_name == "claps":
                logits = model(audio, label, 'test')
            else:
                logits = model(audio)
            loss = criterion(logits, label)

            losses.append(loss.item())

            actual_labels.extend(label.cpu().numpy())
            predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_training(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=50, args=None):
    
    if args.model_name == "moaalm":
        init_epoch(model, train_dataloader, optimizer, criterion, device, 0, args=args)      
    
    for epoch in tqdm(range(epochs), total=epochs):

        train_loss, actual_labels, predicted_labels = run_epoch(model, train_dataloader, optimizer, criterion, device, epoch, args=args)

        if (epoch+1)%5 == 0 or args.model_name == 'pnalm':
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTrain Evaluation (Epoch {epoch + 1}/{epochs})\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, train_loss) 
            

        if (epoch+1)%args.freq_test_model == 0 or epoch == epochs-1 or args.model_name == 'pnalm':
            test_loss, actual_labels, predicted_labels = run_evaluation(model, test_dataloader, criterion, device, args=args)
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTest Evaluation\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, test_loss)

            if ((epoch == epochs-1) or args.model_name == 'pnalm' )and args.do_logging:
                print("\n\nFinal Evaluation")
                print("Saving Results ...")
                save_scores(args.seed, epoch, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
                print("Results Saved\n\n")
                
        if epoch == epochs - 1:
            print("\nTraining Completed!\n")
            save_features_classifier(model, test_dataloader, criterion, device, args=args)
        if args.model_name == 'pnalm' and epoch == 1:
            break
    
    if args.save_model:
        save_model_path = get_save_model_path(args)
        save_model(args, model, save_model_path)
        print(f"Model saved to {save_model_path}")
        
        

def save_features_classifier(model, dataloader, criterion, device, args):
    model.eval()
    save_model_path = 'save_feature/'
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    
    a_list = []  # 存储特征a
    label_list = []  # 存储标签
    our_t_proto_list = []
    our_a_proto_list = []
    org_text_proto_list = []
    with torch.no_grad():
        for i, (audio, label) in enumerate(dataloader):
            print(f"Batch {i+1}/{len(dataloader)}")

            audio = audio.to(device).squeeze(1)
            label = label.to(device)
            
            audio_features, label, text_proto, visual_proto, classnames, org_text_proto = model.save_features(audio, label)
            a_list.append(audio_features.cpu().numpy())
            label_list.append(label.cpu().numpy())
        our_t_proto_list.append(text_proto.cpu().numpy())
        our_a_proto_list.append(visual_proto.cpu().numpy())
        org_text_proto_list.append(org_text_proto.cpu().numpy())

    a_array = np.concatenate(a_list)
    label_array = np.concatenate(label_list)
    our_t_proto_array = np.concatenate(our_t_proto_list)
    our_a_proto_array = np.concatenate(our_a_proto_list)
    org_text_proto_array = np.concatenate(org_text_proto_list)
    
    # 保存特征和标签到文件
    
    np.save(os.path.join(save_model_path, 'our_a_features.npy'), a_array)
    np.save(os.path.join(save_model_path, 'labels.npy'), label_array)
    np.save(os.path.join(save_model_path, 'our_text_prototypes.npy'), our_t_proto_array)
    np.save(os.path.join(save_model_path, 'our_audio_prototypes.npy'), our_a_proto_array)
    np.save(os.path.join(save_model_path, 'org_text_prototypes.npy'), org_text_proto_array)