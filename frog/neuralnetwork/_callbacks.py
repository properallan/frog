import os
import json
import time
import tensorflow as tf
from ray import train
from tensorflow.keras.models import save_model

class PrintCallback(tf.keras.callbacks.Callback):
    """
    Callback para integração com Ray Tune e registro em TensorBoard durante o treinamento com Keras.

    Parâmetros:
    -----------
    log_dir : str
        Caminho para salvar logs do TensorBoard e histórico.
    fr_model : keras.Model, optional
        Modelo funcional externo (por exemplo, com normalizações invertidas) para avaliação fora do fit.
    test_X : np.ndarray, optional
        Conjunto de dados de teste.
    test_y : np.ndarray, optional
        Rótulos/targets reais do conjunto de teste.
    metrics_dict : dict, optional
        Dicionário com nome das métricas como chaves e nomes de funções como valores (str).
        Exemplo: {"R2": "R2", "MAE": "MAE"}
    """
    def __init__(self, log_dir="logs", fr_model=None, test_X=None, test_y=None, metrics_dict=None, model_dir=None):
        super().__init__()
        self.history = {"loss": [], "val_loss": [], "lr": [], "training_iteration": []}
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Pega o learning rate atual (compatível com schedulers)
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except:
            step = self.model.optimizer.iterations
            lr = self.model.optimizer.learning_rate    
            lr = float(tf.keras.backend.get_value(lr(step)))

        logs['lr'] = lr
        
        # Dicionário com métricas a serem reportadas ao Ray Tune
        metrics_to_report = {
            "loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "lr": logs.get("lr"),
            "training_iteration": epoch
        }

        # Registra métricas no TensorBoard
        with self.writer.as_default():
           for key, value in metrics_to_report.items():
               tf.summary.scalar(key, value, step=epoch)
           self.writer.flush()

        # Imprime estatísticas a cada 10 épocas
        if epoch % 1 == 0:
            elapsed = time.time() - self.start_time
            it_s = (epoch + 1) / elapsed
            print(f"Epoch {epoch}: loss={logs['loss']:>4.4f}, val_loss={logs['val_loss']:4>.4f}, lr={lr:>4.4f}, {it_s:>4.4f} it/s")


class TuneReporterCallback(tf.keras.callbacks.Callback):
    """
    Callback para integração com Ray Tune e registro em TensorBoard durante o treinamento com Keras.

    Parâmetros:
    -----------
    log_dir : str
        Caminho para salvar logs do TensorBoard e histórico.
    fr_model : keras.Model, optional
        Modelo funcional externo (por exemplo, com normalizações invertidas) para avaliação fora do fit.
    test_X : np.ndarray, optional
        Conjunto de dados de teste.
    test_y : np.ndarray, optional
        Rótulos/targets reais do conjunto de teste.
    metrics_dict : dict, optional
        Dicionário com nome das métricas como chaves e nomes de funções como valores (str).
        Exemplo: {"R2": "R2", "MAE": "MAE"}
    """
    def __init__(self, log_dir="logs", fr_model=None, test_X=None, test_y=None, metrics_dict=None, model_dir=None, kfold_iteration=None):
        super().__init__()
        self.log_dir = log_dir
        self.history = {"loss": [], "val_loss": [], "lr": [], "training_iteration": []}
        if kfold_iteration is not None:
            self.history.update({"kfold_iteration": []})
        self.writer = tf.summary.create_file_writer(log_dir)
        self.fr_model = fr_model
        self.test_X = test_X
        self.test_y = test_y
        self.metrics_dict = metrics_dict
        self.model_dir = model_dir
        self.epoch = 0
        self.kfold_iteration = kfold_iteration

        #if self.metrics_dict:
        #    self.history.update({k:[] for k,v in self.metrics_dict.items()})

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        from frog.metrics import NRMSE, R2, MAPE, MAXPE, MAE, MSE
        from ray.train import get_context, report

        # Se existir, carrega o histórico anterior
        if os.path.exists(os.path.join(self.log_dir, "history.json")):
            with open(os.path.join(self.log_dir, "history.json"), "r") as f:
                self.history = json.load(f)

        if logs is None:
            logs = {}

        # Pega o learning rate atual (compatível com schedulers)
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except:
            step = self.model.optimizer.iterations
            lr = self.model.optimizer.learning_rate 
            lr = float(tf.keras.backend.get_value(lr(step)))

        logs['lr'] = lr
        
        # Dicionário com métricas a serem reportadas ao Ray Tune
        metrics_to_report = {
            "loss": logs.get("loss"),
            "val_loss": logs.get("val_loss"),
            "lr": logs.get("lr"),
            "training_iteration": epoch,
        }

        if self.kfold_iteration is not None:
            metrics_to_report.update({ "kfold_iteration": self.kfold_iteration})
        

        # Calcula métricas customizadas se os dados forem fornecidos
        if self.metrics_dict is not None and self.test_X is not None and self.test_y is not None and self.fr_model is not None:
            prediction = self.fr_model.predict(self.test_X)
            for key, func_name in self.metrics_dict.items():
                try:
                    metrics_to_report[key] = float(eval(func_name)(self.test_y, prediction))
                except Exception as e:
                    #(f"[Warning] Falha ao calcular a métrica '{key}': {e}")
                    print(f"[Warning] Falha ao calcular a métrica '{key}': {e}")

        # Salva métricas padrão no histórico
        for k,v in self.history.items():
            self.history[k].append(metrics_to_report[k])
        
        # self.history["loss"].append(logs.get("loss"))
        # self.history["val_loss"].append(logs.get("val_loss"))
        # self.history["lr"].append(logs.get("lr"))
        # self.history["training_iteration"].append(epoch)
        
        # Reporta para Ray Tune
        if get_context() and self.kfold_iteration is None:
            report(metrics_to_report)

        # Registra métricas no TensorBoard
        with self.writer.as_default():
           for key, value in metrics_to_report.items():
               tf.summary.scalar(key, value, step=epoch)
           self.writer.flush()

        # Imprime estatísticas a cada 10 épocas
        if epoch % 10 == 0:
            elapsed = time.time() - self.start_time
            it_s = (epoch + 1) / elapsed
            print(f"Epoch {epoch}: loss={logs['loss']:>4.4f}, val_loss={logs['val_loss']:4>.4f}, lr={lr:>4.4f}, {it_s:>4.4f} it/s")

        current_epoch = os.path.join(self.log_dir, "current_epoch.txt")
        with open(current_epoch, "w") as f:
            f.write(f"{epoch+1}")

        #Salva histórico em arquivo JSON
        history_path = os.path.join(self.log_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f)

        if self.model_dir is not None:
                save_model(
                model=self.model,
                filepath=self.model_dir+"/model.keras",
                include_optimizer=True,   # evita problemas com LR schedules customizados
                #save_format="tf"          # força SavedModel (pasta)
            )
        
        self.metrics_to_report = metrics_to_report

    def on_train_end(self, logs=None):
        from frog.metrics import NRMSE, R2, MAPE, MAXPE, MAE, MSE
        from ray.train import get_context, report

        # Calcula métricas customizadas se os dados forem fornecidos
        if self.metrics_dict is not None and self.test_X is not None and self.test_y is not None and self.fr_model is not None:
            prediction = self.fr_model.predict(self.test_X)
            for key, func_name in self.metrics_dict.items():
                try:
                    self.metrics_to_report[key] = float(eval(func_name)(self.test_y, prediction))
                except Exception as e:
                    print(f"[Warning] Falha ao calcular a métrica '{key}': {e}")

        if self.kfold_iteration is not None:
            self.metrics_to_report.update({ "kfold_iteration": self.kfold_iteration})

        # Reporta para Ray Tune
        if get_context():
            report(self.metrics_to_report)

        if self.model_dir is not None:
                save_model(
                model=self.model,
                filepath=self.model_dir+"/model.keras",
                include_optimizer=True,   # evita problemas com LR schedules customizados
                #save_format="tf"          # força SavedModel (pasta)
            )
import math

class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, base_lr=1e-3, total_epochs=100, warmup_epochs=5, final_lr_fraction=0.01):
        super(WarmupCosineDecay, self).__init__()
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.final_lr_fraction = final_lr_fraction

    def on_epoch_begin(self, epoch, logs=None):
        # Cálculo da learning rate por epoch
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            decay_progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            min_lr = self.base_lr * self.final_lr_fraction
            lr = min_lr + (self.base_lr - min_lr) * cosine_decay
        
        # Atualizando o learning rate
        #tf.keras.backend.set_value(self.model.optimizer.learning_rate, float(lr))
        self.model.optimizer.learning_rate = tf.Variable(lr, trainable=False)
        #print(f"Epoch {epoch+1}/{self.total_epochs} - Learning Rate: {lr:.6f}")

from tensorflow.keras.callbacks import ReduceLROnPlateau


class EpochRangeReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, start_epoch=0, end_epoch=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if self.start_epoch <= epoch and (self.end_epoch is None or epoch < self.end_epoch):
            super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if self.start_epoch <= epoch and (self.end_epoch is None or epoch < self.end_epoch):
            super().on_epoch_end(epoch, logs)

import tensorflow as tf

class IncreaseLROnImprovement(tf.keras.callbacks.Callback):
    def __init__(self, factor=2, threshold=0.1, patience=3, monitor="val_loss", start_epoch=0, end_epoch=None):
        super(IncreaseLROnImprovement, self).__init__()
        self.factor = factor  # Multiplicador do LR
        self.threshold = threshold  # Percentual mínimo de queda para considerar melhoria
        self.patience = patience  # Número de épocas consecutivas para aumentar LR
        self.monitor = monitor
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.prev_loss = None
        self.wait = 0  # Contador de épocas consecutivas de melhora

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)

        # Verifica se a época atual está dentro do intervalo especificado
        if epoch < self.start_epoch:
            return  # Ainda não iniciou o intervalo
        if self.end_epoch is not None and epoch >= self.end_epoch:
            return  # Já passou do intervalo

        if current_loss is None:
            return

        if self.prev_loss is not None:
            loss_reduction = (self.prev_loss - current_loss) / self.prev_loss

            if loss_reduction > self.threshold:
                self.wait += 1  # Contar épocas consecutivas de melhora
                if self.wait >= self.patience:
                    old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                    new_lr = old_lr * self.factor
                    self.model.optimizer.learning_rate = tf.Variable(new_lr, trainable=False)
                    print(f"\nAumentando learning rate de {old_lr:.6f} para {new_lr:.6f} (epoch {epoch+1})")
                    self.wait = 0  # Resetar contador após ajuste
            else:
                self.wait = 0  # Resetar contador se não houve melhora suficiente

        self.prev_loss = current_loss


# class IncreaseLROnImprovement(tf.keras.callbacks.Callback):
#     def __init__(self, factor=2, threshold=0.1, patience=3, monitor="val_loss"):
#         super(IncreaseLROnImprovement, self).__init__()
#         self.factor = factor  # Multiplicador do LR
#         self.threshold = threshold  # Percentual mínimo de queda para considerar melhoria
#         self.patience = patience  # Número de épocas consecutivas para aumentar LR
#         self.monitor = monitor
#         self.prev_loss = None
#         self.wait = 0  # Contador de épocas consecutivas de melhora

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         current_loss = logs.get(self.monitor)
        
#         if current_loss is None:
#             return
        
#         if self.prev_loss is not None:
#             loss_reduction = (self.prev_loss - current_loss) / self.prev_loss
            
#             if loss_reduction > self.threshold:
#                 self.wait += 1  # Contar épocas consecutivas de melhora
#                 if self.wait >= self.patience:
#                     old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
#                     new_lr = old_lr * self.factor
#                     tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
#                     #print(f"\nAumentando learning rate de {old_lr:.6f} para {new_lr:.6f} (epoch {epoch+1})")
#                     self.wait = 0  # Resetar contador após ajuste
#             else:
#                 self.wait = 0  # Resetar contador se não houve melhora suficiente

#         self.prev_loss = current_loss