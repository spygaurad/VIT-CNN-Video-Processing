import torch
import torch.optim as optim
import random
import os
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as trF
from tensorboardX import SummaryWriter

from UNet import UNet
from AutoEncoder import AutoEncoder
from Dataloader import CustomDataLoader
from Metrics import JaccardScore, DiceLoss, MixedLoss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
BATCH_SIZE = 64
MODEL_NAME = "IMAGE2IMAGE"
large_file_dir = '/home/spygaurad/vit_video_processing/Transformer-CNN-Hybrid-Network-for-Video-Processing/'


class Model():
 
    def __init__(self, trained=False):
        self.model = AutoEncoder().to(DEVICE)
        # self.jaccard = JaccardScore()

    def psnr(self, reconstructed, original, max_val=1.0): return 20 * torch.log10(max_val / torch.sqrt(F.mse_loss(reconstructed, original)))        


    def train(self, dataset, loss_func, optimizer):

        self.model.train()
        running_loss = 0.0
        running_psnr = 0.0
        counter = 0

        for i, img in tqdm(enumerate(dataset), total=len(dataset)):
            counter += 1
            image = img.to(DEVICE)
            aug_image = image
            if random.random() > 0.5:
                aug_image = trF.hflip(aug_image)
            if random.random() > 0.8:
                aug_image = image + torch.randn(image.size()).to(DEVICE) * 0.05 + 0.0
                # Create 2-5 16x16 blackout patches in the image, along random locations in the axis of height and width
                if random.random() > 0.5:
                    for _ in range(random.randint(0, 3)):
                        x = random.randint(0, image.size(2) - 16)
                        y = random.randint(0, image.size(3) - 16)
                        aug_image[:, :, x:x + 16, y:y + 16] = 0.0

            optimizer.zero_grad()
            output = self.model(aug_image)
            loss = loss_func(output[1], image)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate the Jaccard score here
            psnr = self.psnr(output[1], image)
            running_psnr += psnr.item()


        epoch_loss = running_loss / (counter * BATCH_SIZE)
        epoch_psnr = running_psnr / counter

        return epoch_loss, epoch_psnr




    def validate(self, dataset):

        self.model.eval()
        running_correct = 0.0
        running_psnr = 0.0
        counter = 0

        with torch.no_grad():
            for i, img in tqdm(enumerate(dataset), total=len(dataset)):
                counter += 1
                img = img.to(DEVICE)
                output = self.model(img)

                psnr = self.psnr(output[1], img)
                running_psnr += psnr.item()
    
        epoch_psnr = running_psnr / counter
        return epoch_psnr



    def test(self, dataset, epoch):
        running_psnr = 0.0  
        counter = 0
        num = random.randint(0, len(dataset) // (BATCH_SIZE // 2))

        with torch.no_grad():
            for i, img in tqdm(enumerate(dataset), total=len(dataset)):
                counter += 1
                image = img.to(DEVICE)
                img = image
                for _ in range(random.randint(0, 3)):
                    x = random.randint(0, image.size(2) - 32)
                    y = random.randint(0, image.size(3) - 32)
                    img[:, :, x:x + 32, y:y + 32] = 0.0
                output = self.model(img)
                pred = output[1]
                psnr = self.psnr(output[1], image)  
                running_psnr += psnr.item()

                if i == num:
                    try:
                        os.makedirs(f"saved_samples/{MODEL_NAME}", exist_ok=True)
                    except:
                        pass
                    image = img[0].cpu().numpy().transpose((1, 2, 0))
                    pred = pred[0].cpu().numpy().transpose((1, 2, 0))
                    image = (image * 255).astype('uint8')
                    pred = (pred * 255).astype('uint8')
                    image_pil = Image.fromarray(image)
                    pred_pil = Image.fromarray(pred)
                    # image_pil.save(f"saved_samples/{MODEL_NAME}/{epoch}_image.jpg")
                    # pred_pil.save(f"saved_samples/{MODEL_NAME}/{epoch}_pred.jpg")
                    stacked_image = Image.new('RGB', (image_pil.width * 2, image_pil.height))
                    stacked_image.paste(image_pil, (0, 0))
                    stacked_image.paste(pred_pil, (image_pil.width, 0))
                    stacked_image.save("stacked_image.jpg")
                    stacked_image.save(f"saved_samples/{MODEL_NAME}/{epoch}.jpg")

        epoch_psnr = running_psnr / counter 
        return epoch_psnr



 
    def fit(self, epochs, lr):

        print(f"Using {DEVICE} device...")
        print("Loading Datasets...")
        train_data, val_data, test_data = CustomDataLoader(BATCH_SIZE).get_data()
        print("Dataset Loaded.")

        print("Initializing Parameters...")
        self.model = self.model.to(DEVICE)
        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total parameters of the model is: {:.2f}{}".format(total_params / 10**(3 * min(len(str(total_params)) // 3, len(["", "K", "M", "B", "T"]) - 1)), ["", "K", "M", "B", "T"][min(len(str(total_params)) // 3, len(["", "K", "M", "B", "T"]) - 1)]))

        print(f"Initializing the Optimizer")
        optimizer = optim.AdamW(self.model.parameters(), lr)
        print(f"Beginning to train...")

        mixedloss = MixedLoss(0.5, 0.5)
        # l2loss = torch.nn.MSELoss()

        val_psnr_epochs = []
        writer = SummaryWriter(f'runs/{MODEL_NAME}/')
        os.makedirs(f"{large_file_dir}checkpoints/", exist_ok=True)
        os.makedirs(f"{large_file_dir}saved_model/", exist_ok=True)


        for epoch in range(1, epochs+1):

            print(f"Epoch No: {epoch}")

            train_loss, train_psnr = self.train(dataset=train_data, loss_func=mixedloss, optimizer=optimizer)

            val_psnr = self.validate(dataset=val_data)
            val_psnr_epochs.append(val_psnr)

            print(f"Train Loss:{train_loss}, Train PSNR:{train_psnr}, Validation PSNR:{val_psnr}")

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("PSNR/Train", train_psnr, epoch)
            writer.add_scalar("PSNR/Val", val_psnr, epoch)


            if max(val_psnr_epochs) == val_psnr:
                torch.save(self.model.state_dict(), f"{large_file_dir}checkpoints/{MODEL_NAME}.pth")
            
            if epoch%5==0:
                print("Saving model")
                torch.save(self.model.state_dict(), f"{large_file_dir}saved_model/{MODEL_NAME}_{epoch}.pth")
                test_psnr = self.test(test_data, epoch)
                writer.add_scalar("PSNR/Test", test_psnr)
                print("Model Saved")

    
            print("Epoch Completed. Proceeding to next epoch...")

        print(f"Training Completed for {epochs} epochs.")


    def infer_a_random_sample(self):
        
        try:
            os.makedirs(f"test_samples/{MODEL_NAME}", exist_ok=True)
        except:
            pass
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])



model = Model()
model.fit(250, 1e-3)