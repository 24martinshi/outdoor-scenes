from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset_waymo import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

def main():
    # Configs
    # resume_path = './models/control_sd15_ini_waymo.ckpt'
    # resume_path = './models/control_sd15_ini_waymo_seg.ckpt'
    # resume_path = './models/control_sd15_ini.ckpt'
    resume_path = './models/control_sd15_seg-Copy1.pth'  

    # TODO: modifying batch size to reduce GPU memory required
    batch_size = 4
    # batch_size = 1

    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control


    # Misc
    dataset = MyDataset()

    # TODO: slowly increase num_workers to speed up training
    dataloader = DataLoader(dataset, num_workers=16, batch_size=batch_size, shuffle=True)
    # dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)

    checkpoint_callback_last = ModelCheckpoint(

        # TODO: adjust these every run!
        dirpath='./models/checkpoints/full_unif_dataset/',
        filename='checkpoint-{epoch}',
        save_top_k=-1,
        # every_n_train_steps=1000,
        every_n_epochs=1,
        save_last=True,  
    )
    
    # TODO: changing mixed precision values, gpu values
    # trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
    # trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger])
    
    # trainer = pl.Trainer(max_epochs=2, gpus=1, precision=16, callbacks=[logger])
    trainer = pl.Trainer(max_epochs=8, gpus=1, precision=16, callbacks=[logger, checkpoint_callback_last])
    
    # trainer = pl.Trainer(gpus=2, strategy='ddp', accumulate_grad_batches=1, precision=32, callbacks=[logger])
    # trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger])


    # Train!
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
    
