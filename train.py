import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_f1 = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target, sequence_lengths = batch.text[0], batch.label, batch.text[1]

            target.data.sub_(1) # Done to align the labels [1,2,3] -> [0,1,2]
            
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            if args.rnn:
                logit = model(feature, sequence_lengths)
            else:
                logit = model(feature)
            
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 

                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_f1 = eval(dev_iter, model, args)
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                model.train()
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    macro_f1_sum = 0
    correct_labels = []
    predicted_labels = []
    for batch in data_iter:
        feature, target, sequence_lengths = batch.text[0], batch.label, batch.text[1]
        target.data.sub_(1)

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        if args.rnn:
            logit = model(feature, sequence_lengths)
        else:
            logit = model(feature)

        loss = F.cross_entropy(logit, target, reduction = 'sum')

        avg_loss += loss.item()
        predict_labels = torch.max(logit, 1)[1].view(target.size()).data
        corrects += (predict_labels == target.data).sum()
        macro_f1_sum += f1_score(predict_labels.cpu().numpy(), target.data.cpu().numpy(), average="macro")
        correct_labels += list(target.data.cpu().numpy())
        predicted_labels += list(predict_labels.cpu().numpy())

    if args.test:
    	print(classification_report(correct_labels, predicted_labels, digits = 4))
        
    size = len(data_iter.dataset)
    batches = len(data_iter)
    avg_loss /= size
    macro_f1 = macro_f1_sum/batches
    accuracy = 100.0 * corrects/size
	
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})  f1: {} \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size, macro_f1))
    return macro_f1




def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model, save_path)
