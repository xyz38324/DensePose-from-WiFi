from detectron2.engine import SimpleTrainer
import time
import torch
class CustomTrainer(SimpleTrainer):
    def __init__(self,cfg, model, data_loader, optimizer):
        super().__init__(model, data_loader, optimizer)
        self.loss_cls = cfg.LOSS.cls
        self.loss_box_reg = cfg.LOSS.box
        self.loss_keypoint = cfg.LOSS.kp
        self.loss_transfer = cfg.LOSS.tr
        self.loss_densepose = cfg.LOSS.dp


    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        
        loss_dict["loss_cls"] *= self.loss_cls
        loss_dict["loss_box_reg"] *= self.loss_box_reg
        loss_dict["loss_keypoint"] *= self.loss_keypoint
        loss_dict["loss_transfer"] *= self.loss_transfer
        loss_dict["loss_densepose"] = self.loss_densepose*(loss_dict['loss_densepose_U']+loss_dict['loss_densepose_V']
                                                            +loss_dict['loss_densepose_I']+loss_dict['loss_densepose_S'])
        del loss_dict['loss_densepose_U']
        del loss_dict['loss_densepose_V']
        del loss_dict['loss_densepose_I']
        del loss_dict['loss_densepose_S']
        
        losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()
        losses.backward()

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)

       
        self.optimizer.step()

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj
