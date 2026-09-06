
import os
import sys
import json
import inspect
import importlib

#---->
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torchmetrics
from torchmetrics.regression import ( PearsonCorrCoef, MeanAbsoluteError)  

#---->
import pytorch_lightning as pl

from core.model_adapters import get_adapter

class BaseModule(pl.LightningModule):

    #---->init
    def __init__(self, **kwargs):
        super(BaseModule, self).__init__()

        self.config = kwargs['cfg']
        self.model_config = self.config.MODEL
        
        self.file_name, self.class_name = self.model_config.model_name.split('.')
        self.adapter = get_adapter(self.class_name, self.model_config.get("adapter", None))

        num_outputs = self.config.DATA.num_outputs
        target = self.config.TRAINING.monitor
        
        if self.config.DATA.mode == 'cv':
            self.save_hyperparameters(ignore=["cfg"])
            
        if self.config.DATA.mode == 'inference':
            self.predictions = []
        
        self.load_model()

        metrics = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = num_outputs),
                                                MeanAbsoluteError(num_outputs = num_outputs)
                                                ])
        self.test_metrics = metrics.clone(prefix = 'test_')        
        if target:
            metrics['target'] = metrics.pop(target)
            idx_target = {v[0]: k for k,v in metrics.compute_groups.items()}[target]
            metrics.compute_groups[idx_target] = ['target']
        self.valid_metrics = metrics.clone(prefix = 'val_')
        
        if os.path.exists(f"{self.config.DATA.output_path}/idx_top.npy"):
            num_hpg = np.load(f"{self.config.DATA.output_path}/idx_top.npy").shape[0]
            self.test_metrics_hpg = torchmetrics.MetricCollection([PearsonCorrCoef(num_outputs = num_hpg),
                                                    MeanAbsoluteError(num_outputs = num_hpg)
                                                    ]).clone(prefix = 'test_', postfix='_hpg')
        
        self.avg_pcc = torch.zeros(num_outputs)
        
        if not self.config.GENERAL.save_predictions:
            self.preds = {}
        
        
    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def training_step(self, batch, batch_idx):
        batch = self.adapter.prepare_batch(self, batch, stage='train')
        results_dict = self.adapter.forward(self, batch, phase='train')
        
        #---->Loss
        loss = results_dict['loss']        
        self.log("train_loss", loss)
        
        return {'loss': loss}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(f"Parameter {name} is unused (no grad)")
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.adapter.after_optimizer_step(self)

    def _slice_gene_outputs(self, logits):
        """Slice a fixed-width model output down to the genes actually
        needed for this run — either an external evaluation against a
        dataset that doesn't measure the model's full training gene panel
        (see _external_gene_overlap in api/stpbench.py), or a user-supplied
        predict(gene_list=...) restriction (see _user_gene_overlap).

        Some adapters (zero-shot models like DeepSpotM/STPath) already
        narrow their own output to the requested gene subset inside
        forward() itself, via dataset.genes — so `logits` may already be
        the target width by the time it gets here. Slicing again would
        apply gene_output_indices (absolute positions into the *full*
        training panel) to an already-narrow tensor, which is wrong at
        best and an out-of-range index_select at worst. Compare against
        the target width directly rather than inferring "already sliced"
        from the presence of `label` (predict_step has none)."""
        gene_output_indices = self.config.DATA.get('gene_output_indices')
        if gene_output_indices is None or logits.shape[-1] == len(gene_output_indices):
            return logits
        # gene_output_indices is fixed for the whole run — cache the index
        # tensor instead of rebuilding it from a Python list on every
        # batch (this is called from every train/val/test/predict step).
        idx = getattr(self, '_gene_output_idx', None)
        if idx is None or idx.device != logits.device:
            idx = torch.as_tensor(gene_output_indices, dtype=torch.long, device=logits.device)
            self._gene_output_idx = idx
        return logits.index_select(-1, idx)

    def validation_step(self, batch, batch_idx):
        batch = self.adapter.prepare_batch(self, batch, stage='val')
        results_dict = self.adapter.forward(self, batch, phase='val')
        label = self.adapter.get_label(self, batch, results_dict, stage='val')

        #---->Loss
        if 'logits' in results_dict:
            logits = self._slice_gene_outputs(results_dict['logits'])

            val_metric = self.valid_metrics(logits, label)
            val_metric = {k: v.nanmean() if len(v.shape) > 0 else v for k, v in val_metric.items()}
            # val_metric["val_target"] = torch.nan_to_num(val_metric["val_target"], nan=0.0)
            
            self.log_dict(val_metric, on_epoch = True, logger = True, sync_dist=True, batch_size = 1)
            outputs = {'logits': logits, 'label': label}
        else:
            loss = results_dict['loss']
            self.log_dict({'val_target': loss}, on_epoch = True, logger = True, sync_dist=True, batch_size = 1)
            outputs = {'loss': loss}
        
        return outputs

    def test_step(self, batch, batch_idx):
        batch = self.adapter.prepare_batch(self, batch, stage='test')
        results_dict = self.adapter.forward(self, batch, phase='test')
        label = self.adapter.get_label(self, batch, results_dict, stage='test')
            
        
        #---->Loss
        logits = self._slice_gene_outputs(results_dict['logits'])
        # label = batch['label']

        test_metric = self.test_metrics(logits, label)
        if os.path.exists(f"{self.config.DATA.output_path}/idx_top.npy"):
            idx_top = np.load(f"{self.config.DATA.output_path}/idx_top.npy")
            idx_top = torch.tensor(idx_top).to(logits.device)
            test_metric_hpg = self.test_metrics_hpg(logits[:, idx_top], label[:, idx_top])
            test_metric_hpg = {k: v.nanmean() if len(v.shape) > 0 else v for k, v in test_metric_hpg.items()}
            test_metric_hpg['epoch'] = self.step_epoch
            self.log_dict(test_metric_hpg, 
                          on_epoch = True, 
                          logger=True,
                          sync_dist=True, 
                          batch_size = 1)
        else:
            self.avg_pcc += test_metric['test_PearsonCorrCoef'].cpu()
            
        test_metric = {k: v.nanmean() if len(v.shape) > 0 else v for k, v in test_metric.items()}
        epoch = getattr(self, "step_epoch", None)
        if epoch is not None:
            test_metric['epoch'] = epoch
        
        self.log_dict(test_metric, on_epoch = True, logger = True, sync_dist=True, batch_size = 1)
        
        # _id = dataset.int2id[batch_idx]
        
        if self.config.GENERAL.save_predictions:
            self.save_predictions(logits, batch_idx)

        else:
            self.preds[batch_idx] = logits

        outputs = {'logits': logits, 'label': label}
        
        return outputs

    # def on_test_epoch_end(self):
    #     parent_dir = "/".join(self.config.DATA.pred_path.split('/')[:-1])
    #     if not os.path.exists(f"{parent_dir}/idx_top.npy"):
    #         print("on_test_epoch_end")
    #         pcc_rank = torch.argsort(torch.argsort(self.avg_pcc, dim=-1), dim=-1) + 1
    #         np.save(f"{self.config.DATA.pred_path}/pcc_rank.npy", pcc_rank.numpy())
    
    def predict_step(self, batch, batch_idx):
        dataset = self.adapter.get_predict_dataset(self)
        # _id = dataset.int2id[batch_idx]
        _id = self.config.DATA.data_id
        
        batch = self.adapter.prepare_predict_batch(self, batch, dataset)
        results_dict = self.adapter.forward(self, batch, phase='test')

        #---->Loss
        pred = self._slice_gene_outputs(results_dict['logits'])

        self.predictions.append(pred)
        
        return pred, _id
    
    def on_predict_epoch_end(self):
        preds = torch.cat(self.predictions, 0)
        self.save_predictions(preds)
        
        self.predictions = []

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.TRAINING.learning_rate)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, 
    #         mode=self.config.TRAINING.mode,
    #         factor=self.config.TRAINING.lr_scheduler.factor,
    #         patience=self.config.TRAINING.lr_scheduler.patience
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": 'val_target'
    #         }
    #     }
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.TRAINING.learning_rate)
        if self.config.TRAINING.mode == 'max':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.TRAINING.mode,
                factor=self.config.TRAINING.lr_scheduler.factor,
                patience=self.config.TRAINING.lr_scheduler.patience
            )
        elif self.config.TRAINING.mode == 'min':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
        else:
            raise ValueError(f"Invalid mode: {self.config.TRAINING.mode}. Must be 'max' or 'min'.")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": 'val_target'
            }
        }
        
    def save_predictions(self, preds, batch_idx=None):
        preds = preds.detach().cpu().numpy().astype(np.float32)
        
        if self.config.DATA.mode == 'inference':
            name, genes, coords = self.adapter.inference_prediction_context(self)

            gene_type = self.config.DATA.gene_type
            num_genes = self.config.DATA.num_genes
            
            if genes is None or len(genes) == 0:
                if num_genes == 30:
                    dataset = self.config.DATA.ref_data_dir.split('/')[-1]
                    data_dir = self.config.DATA.ref_data_dir.replace(f'/{dataset}', '')
                    gene_path = f"{data_dir}/ncche-visium_hest-PRAD_wustl-BRCA_30genes.json"
                else:    
                    gene_path = f"{self.config.DATA.ref_data_dir}/{gene_type}_{num_genes}genes.json"

                try:
                    with open(gene_path, 'r') as f:
                        genes = json.load(f)['genes']
                except:
                    raise ValueError('Gene list file for prediction not found!')
            
            # output_path = f"{self.config.DATA.pred_path}/{name}.h5ad"
            adata_pred = sc.AnnData(
                X=preds,
                var=pd.DataFrame(index=genes)
            )
            if coords is not None:
                # suppress_library_output() (see api/output_control.py)
                # redirects sys.stdout for the whole predict run and
                # silences UserWarning, so warnings.warn()/print() here
                # would vanish -- write straight to the real stderr instead.
                if len(coords) == adata_pred.n_obs:
                    adata_pred.obsm['spatial'] = np.asarray(coords)
                else:
                    print(
                        f"[STPBench] WARNING: spatial coords count ({len(coords)}) "
                        f"does not match prediction count ({adata_pred.n_obs}) for "
                        f"{name}; skipping obsm['spatial'].",
                        file=sys.stderr,
                    )
        else:
            name, genes, id2dir = self.adapter.evaluation_prediction_context(self, batch_idx)
            if isinstance(genes, dict):
                data_dir = id2dir[name]
                genes = genes[data_dir]
            
            # output_path = f"{self.config.DATA.output_path}/{name}.h5ad"
            # output_path = f"{self.config.DATA.pred_path}/{name}.h5ad"

            data_dir = self.config.DATA.data_dir
            st_dir = f"{data_dir}/st"
            if not os.path.isdir(st_dir):
                st_dir = f"{data_dir}/adata"
            label_path = f"{st_dir}/{name}.h5ad"
            if not os.path.isfile(label_path) and st_dir.endswith("/st"):
                fallback = f"{data_dir}/adata/{name}.h5ad"
                if os.path.isfile(fallback):
                    label_path = fallback
            label = sc.read_h5ad(label_path)
            label = label[:, genes]
            adata_pred = label.copy()
            del adata_pred.X
            adata_pred.X = preds
        
        output_path = f"{self.config.DATA.pred_path_fold}/{name}.h5ad"
        # HEST h5ad files store strings (obs index, var columns, …) as
        # pandas ArrowStringArray, which anndata has no registered h5ad writer for.
        # Setting adata.obs/var via the property setter can re-convert them, so
        # write through a fresh AnnData built from plain numpy object arrays.
        def _native_df(df: pd.DataFrame) -> pd.DataFrame:
            cols = {}
            for c in df.columns:
                s = df[c]
                if isinstance(s.dtype, np.dtype):
                    cols[c] = s.values
                else:
                    try:
                        # Numeric Arrow dtypes (int, float, bool) expose .numpy_dtype;
                        # convert with it to preserve int64/float64 — object arrays of
                        # ints would be misidentified as vlen-string by anndata's writer.
                        cols[c] = s.to_numpy(dtype=s.dtype.numpy_dtype, na_value=0)
                    except (AttributeError, NotImplementedError):
                        # String / other non-numeric extension types → object array.
                        cols[c] = s.to_numpy(dtype=object, na_value=None)
                    except Exception:
                        cols[c] = np.array(s.tolist(), dtype=object)
            result = pd.DataFrame(cols, index=pd.Index(df.index.tolist(), dtype=object))
            # pandas 3.0+ re-infers object arrays of strings as StringDtype
            # (Arrow-backed) during DataFrame construction; cast them back to object.
            for c in list(result.columns):
                if not isinstance(result[c].dtype, np.dtype):
                    result[c] = result[c].astype(object)
            return result
        sc.AnnData(
            X=adata_pred.X,
            obs=_native_df(adata_pred.obs),
            var=_native_df(adata_pred.var),
            obsm=dict(adata_pred.obsm),
        ).write(output_path)
    
    def load_model(self):
        try:
            Model = getattr(importlib.import_module(
                f'model.{self.file_name}'), self.class_name)
        except Exception as exc:
            raise ValueError(
                f"Invalid Module File Name or Invalid Class Name: "
                f"{self.model_config.model_name}"
            ) from exc
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.model_config dictionary. You can also input any args
            to overwrite the corresponding value in self.model_config.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.model_config.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.model_config, arg)
        args1.update(other_args)
        return Model(**args1)
