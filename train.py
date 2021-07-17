import torch
import time
from torch._C import device
from torch.optim import lr_scheduler
from model.utils import create_look_ahead_mask, create_padding_mask
from model.transformer import Transformer
from hparameters import hparameters
from dataloaders.dataloader import CreateDataLoaders
import pandas as pd
import copy
import os

class CustomSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps =warmup_steps

        super(CustomSchedule, self).__init__(optimizer)

    def get_lr(self):
        # print('*'*27, self._step_count)
        arg1 = self._step_count ** (-0.5)
        arg2 = self._step_count * (self.warmup_steps ** -1.5)
        dynamic_lr = (self.d_model ** (-0.5)) * min(arg1, arg2)
        # print('dynamic_lr:', dynamic_lr)
        return [dynamic_lr for group in self.optimizer.param_groups]

def mask_accuracy_func(real, pred, pad=1):
    pred = pred.argmax(dim=-1)
    corrects = pred.eq(real)

    mask = torch.logical_not(real.eq(pad))

    corrects *= mask

    return corrects.sum().float() / mask.sum().item()

def mask_loss_function(real, pred, pad=1):
    loss_object = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_object(pred.transpose(-1, -2), real)

    mask = real.not_equal(pad)

    loss *= mask

    return loss.sum() / mask.sum().item()

def create_mask(inp, target):
    enc_padding_mask = create_padding_mask(inp, pad=1)
    dec_padding_mask = create_padding_mask(inp, pad=1)

    look_ahead_mask = create_look_ahead_mask(target.shape[1])
    dec_target_padding_mask = create_padding_mask(target, pad=1)
    combined_mask = torch.max(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
    
def validate_step(model, inp, targ, device):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp)

    inp = inp.to(device)
    targ_inp = targ_inp.to(device)
    targ_real = targ_real.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    combined_mask = combined_mask.to(device)
    dec_padding_mask = dec_padding_mask.to(device)

    model.eval()  # 设置eval mode

    with torch.no_grad():
        # forward
        prediction, _ = model(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
        # [b, targ_seq_len, target_vocab_size]
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

        val_loss = mask_loss_function(targ_real, prediction)
        val_metric = mask_accuracy_func(targ_real, prediction)

    return val_loss.item(), val_metric.item()


def train_model():
    use_cuda = torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'

    create_dataloader = CreateDataLoaders(hparameters['training_dataset'])
    train_dataloader, val_dataloader = create_dataloader.build_data_loader()

    transformer = Transformer(num_layers=hparameters['num_layers'],
                              d_model=hparameters['d_model'],
                              num_heads=hparameters['num_heads'],
                              d_feedforward=hparameters['dff'],
                              input_vocab_size=len(create_dataloader.src_text.vocab),
                              target_vocab_size=len(create_dataloader.targ_text.vocab),
                              pe_input=hparameters['input_vocab_size'],
                              pe_target=hparameters['target_vocab_size'])
    transformer = transformer.to(device)
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = CustomSchedule(optimizer, 
                                  hparameters['d_model'], 
                                  warmup_steps=4000)

    df_history = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'val_loss', 'val_acc'])

    start_time = time.time()
    print('#'*50)
    print("start training...")

    best_acc = 0.

    for epoch in range(1, hparameters['training_epochs'] + 1):
        loss_sum = 0.
        metric_sum = 0.

        for step, (input, label) in enumerate(train_dataloader, start=1):
            label_inp = label[:, :-1]
            label_real = label[:, 1:]
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(input, label_inp)

            input = input.to(device)
            label_inp = label_inp.to(device)
            label_real = label_real.to(device)
            enc_padding_mask = enc_padding_mask.to(device)
            combined_mask = combined_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)

            transformer.train()

            optimizer.zero_grad()

            prediction, _ = transformer(input, label_inp, enc_padding_mask, 
                    combined_mask, dec_padding_mask)
            
            loss = mask_loss_function(label_real, prediction)
            metric = mask_accuracy_func(label_real, prediction)

            loss.backward()
            optimizer.step()
        
            loss_sum += loss.item()
            metric_sum += metric.item()

            if (step % hparameters['log_every_step'] ==  0) :
                print('*' * 8, f'[step = {step}] loss: {loss_sum / step:.3f}, acc: {metric_sum / step:.3f}')
            
            lr_scheduler.step()
        
        val_loss_sum = 0.
        val_metric_sum = 0.
        for val_step, (inp, targ) in enumerate(val_dataloader, start=1):
            # inp [64, 10] , targ [64, 10]
            loss, metric = validate_step(transformer, inp, targ, device)

            val_loss_sum += loss
            val_metric_sum += metric

        record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
        df_history.loc[epoch - 1] = record
        
        print('EPOCH = {} loss: {:.3f}, acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}'.format(
            record[0], record[1], record[2], record[3], record[4]))

        current_acc_avg = val_metric_sum / val_step
        if current_acc_avg > best_acc:  # 保存更好的模型
            best_acc = current_acc_avg
            checkpoint = hparameters["checkpoint_path"] + '{:03d}_{:.2f}_ckpt.tar'.format(epoch, current_acc_avg)
            if not os.path.exists(hparameters["checkpoint_path"]):
                os.makedirs(hparameters["checkpoint_path"]) 
            model_sd = copy.deepcopy(transformer.state_dict())
            torch.save({
                'loss': loss_sum / step,
                'epoch': epoch,
                'net': model_sd,
                'opt': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, checkpoint)

    print('finishing training...')
    endtime = time.time()
    time_elapsed = endtime - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == "__main__":
    train_model()

