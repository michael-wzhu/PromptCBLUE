import os
import sys
import time
import torch
import numpy as np
import bmtrain as bmt
import distutils.version  # noqa: F401

from tensorboardX import SummaryWriter


sys.path.insert(0, "./")
from medical_prompts.src.finetune_bloom_bmt.data_utils import pad
from medical_prompts.src.finetune_bloom_bmt.dataset import JsonlDataset, DistributedDataLoader


def pad_collate_fn():
    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    "The elements of the batch contain different keys."
                    f"Cannot batch them ({set(item.keys())} != {keys})"
                )
        padded = {}
        for key in keys:
            if key in ["target", "label", "labels"]:
                _padding_value = -100
            else:
                _padding_value = 0
            padded[key] = pad(items, key, _padding_value, padding_side="right")
        return padded

    return inner


class CPMAntPlusTune:
    def __init__(
        self,
        model,
        tokenizer,
        lr=5e-3,
        warmup_iters=50,
        task_id=2,
        max_len=256,
        cls_num=None,
        epochs=1,
        batch_size=1,
        num_workers=1,
        eval_interval=50,
        output_path="output",
        early_stop_patience=None,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = bmt.optim.AdamOffloadOptimizer(
            model.parameters(), weight_decay=0.001
        )
        self.lr_scheduler = bmt.lr_scheduler.Linear(
            self.optimizer,
            start_lr=lr,
            warmup_iter=warmup_iters,
            end_iter=-1,
            num_iter=0,
        )
        self.optim_manager = bmt.optim.OptimManager(loss_scale=2048)
        self.optim_manager.add_optimizer(self.optimizer, self.lr_scheduler)

        self.loss_function = bmt.loss.FusedCrossEntropy(ignore_index=-100)
        self.task_id = task_id
        self.max_len = max_len
        self.cls_num = cls_num
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.num_workers = num_workers
        self.batch_size = batch_size
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path
        self.early_stop_patience = early_stop_patience

        tensorboard_log_path = os.path.join(output_path, "logs")
        if bmt.rank() == 0 and tensorboard_log_path is not None:
            self.summary_writer = SummaryWriter(log_dir=tensorboard_log_path)
        else:
            self.summary_writer = None

    def _ensure_tensor_on_device(self, inputs, device):

        if isinstance(inputs, dict):
            return {
                name: self._ensure_tensor_on_device(tensor, device)
                for name, tensor in inputs.items()
            }
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype == torch.float16:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def process_data(self, inputs, **kwargs):
        raise NotImplementedError("process_data is not implemented")

    def _forward(self, inputs, **kwargs):
        raise NotImplementedError("_forward is not implemented")

    def forward(self, train_dataloader, eval_dataloader, cls_num=None):

        average_time = 0
        average_time_shift = 0.9
        global_step = 0
        best_eval_loss = 1e9
        best_eval_step = 0

        self.optimizer.zero_grad()
        for epoch in range(self.epochs):
            for idx, train_data in enumerate(train_dataloader):
                train_data = self._ensure_tensor_on_device(train_data, device="cuda")
                self.model.train()
                global_step += 1

                start_time = time.time()
                # custom part for different models

                loss = self._forward(train_data, cls_num=cls_num)
                global_loss = bmt.sum_loss(loss).item()
                self.optim_manager.backward(loss)

                grad_norm = self.optim_manager.clip_grad_norm(
                    self.optimizer.param_groups,
                    max_norm=2.0,
                    # scale=self.optimizer.scale,
                    norm_type=2,
                )
                self.optim_manager.step()
                self.optim_manager.zero_grad()

                iteration_time = time.time() - start_time
                average_time = (
                    average_time * average_time_shift + (1 - average_time_shift) * iteration_time
                )

                bmt.print_rank(
                    "| Train | Epoch: {:3d} | Iter: {:6d} | loss: {:.4f} |"
                    "lr: {:.4e} | time: {:.4f} | grad_norm: {:.4f}".format(
                        epoch,
                        idx,
                        global_loss,
                        self.lr_scheduler.current_lr,
                        average_time / (1 - pow(average_time_shift, global_step + 1)),
                        grad_norm,
                    )
                )

                if bmt.rank() == 0 and self.summary_writer is not None:
                    self.summary_writer.add_scalar("Loss/train", global_loss, global_step)

                self.optimizer.zero_grad()

                if global_step % self.eval_interval == 0:
                    self.model.eval()

                    total_loss = 0
                    cnt = 0
                    with torch.inference_mode():
                        for eval_data in eval_dataloader:
                            cnt += 1
                            bmt.print_rank(f"dev set step: {cnt}")

                            eval_data = self._ensure_tensor_on_device(eval_data, device="cuda")
                            loss = self._forward(eval_data)
                            total_loss += bmt.sum_loss(loss).item()

                    assert cnt == len(eval_dataloader)
                    eval_loss = total_loss / cnt

                    if bmt.rank() == 0 and self.summary_writer is not None:
                        self.summary_writer.add_scalar("Loss/eval", eval_loss, global_step)

                    bmt.print_rank(
                        "| Eval | Iter: {:6d} | loss: {:.4f}".format(global_step, eval_loss)
                    )

                    # save best model
                    if eval_loss < best_eval_loss:
                        bmt.print_rank(
                            "[INFO] Iteration {} is the best checkpoint now!".format(global_step)
                        )
                        best_eval_loss = eval_loss
                        best_eval_step = global_step
                        ckpt_full_path = os.path.join(self.output_path, "best.pt")

                        state_dict = self.model.state_dict()
                        if bmt.rank() == 0:
                            torch.save(state_dict, ckpt_full_path)
                    elif (
                        self.early_stop_patience is not None
                        and (global_step - best_eval_step) // self.eval_interval
                        >= self.early_stop_patience
                    ):
                        bmt.print_rank("[INFO] Early stop at iteration {}!".format(global_step))
                        return
            bmt.print_rank(f"[INFO] Epoch {epoch} finished!")
        return

    def run(self, inputs):
        collate_fn = pad_collate_fn()

        train_dataset = JsonlDataset(inputs["train"], self.process_data)
        train_dataloader = DistributedDataLoader(
            train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

        eval_dataset = JsonlDataset(inputs["dev"], self.process_data)
        eval_dataloader = DistributedDataLoader(
            eval_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

        self.forward(train_dataloader, eval_dataloader, cls_num=self.cls_num)


class CPMAntPlusNLGTune(CPMAntPlusTune):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.truncate_num = 0

    def process_data(self, inputs):
        res = {}
        target = inputs["target"]
        input = inputs["input"]
        if "答：" in input:
            prompt = "[Round 0]\n问：{}".format(input)
        else:
            prompt = "[Round 0]\n问：{}\n答：".format(input.strip())

        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt)
        target_ids = self.tokenizer.encode(target) + [self.tokenizer.eos_token_id]

        if len(input_ids) + len(target_ids) > self.max_len:
            self.truncate_num += 1
            if self.truncate_num % 100 == 0:
                bmt.print_rank(
                    f"[Warning] There are more than {self.truncate_num} instances are truncated!"
                    "Consider to increase max_len!"
                )

            tr_input_length = self.max_len - len(target_ids)
            if tr_input_length > 0:
                input_ids = input_ids[-tr_input_length:]
            else:
                # target is too long
                bmt.print_rank(
                    f"[Warning] target {target} length exceeds max_len, check your data!"
                )
                input_ids = []
                target_ids = target_ids[-(self.max_len) :]

        res["input_ids"] = input_ids + target_ids
        res["labels"] = [-100] * len(input_ids) + target_ids
        for key in res:
            res[key] = torch.tensor(res[key]).int().unsqueeze(0)

        return res

    def _forward(self, model_inputs, **kwargs):
        labels = model_inputs.pop("labels", None)
        res = self.model(**model_inputs)
        logits = res.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_function(shift_logits.view(-1, logits.shape[-1]), shift_labels.view(-1))

        return loss