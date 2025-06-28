import pytorch_lightning as pl
import torch


class YolosForSignatureDetection(pl.LightningModule):
    """PyTorch Lightning module for YOLOS model fine-tuning on signature detection."""

    def __init__(self, lr, weight_decay, model, train_dl, val_dl):
        """Initialize the YOLOS model for signature detection fine-tuning."""
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_dl = train_dl
        self.val_dl = val_dl

    def forward(self, pixel_values):
        """Forward pass through the model."""
        return self.model(pixel_values=pixel_values)

    def common_step(self, batch, batch_idx):
        """Common step for training and validation."""
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        # Clear CUDA cache
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for the model."""
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        # Clear CUDA cache
        torch.cuda.empty_cache()
        return loss

    def configure_optimizers(self):
        """Configure the optimizer for the model."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        """Get the training dataloader."""
        return self.train_dl

    def val_dataloader(self):
        """Get the validation dataloader."""
        return self.val_dl
