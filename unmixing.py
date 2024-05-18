import numpy as np
import torch
import gc
from torch.nn.functional import cosine_similarity

class LinearUnmixing:
    def __init__(self,
                 homogeneous: torch.Tensor,
                 observed: torch.Tensor = None,
                 loss: callable = torch.nn.L1Loss(reduction='sum'),
                 optimizer: callable = torch.optim.Adam,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 optimizer_kwrgs: dict = {'lr': 0.4},
                 base_size:int = None
                 ):

        self.device = device
        self.homogeneous = homogeneous.to(torch.float32).to(device)
        self.observed = observed if observed is None else observed.to(device)
        self.base_size = base_size
        self.got_null = None
        self.keep_values = None
        self.observed = None
        self.set_observed(observed)

        self.loss = loss.to(device) if type(loss) == torch.nn.modules.loss.L1Loss else loss
        self.optimizer = optimizer
        self.optimizer_kwrgs = optimizer_kwrgs

    def _clear_gpu_memory(self):
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

    def _delete_all_errors(self):
        self.rates = None
        self.predictions = None
        self.errors = None
        self._clear_gpu_memory()

    def _keep_best_root_rates(self,
                              new_errors,
                              new_root_rates):
        if self.errors is None:
            self.errors = new_errors
            self.root_rates = new_root_rates
        else:
            with torch.no_grad():
                new_errors = torch.reshape(new_errors,(new_errors.shape[0], 1))
                current_is_better = new_errors < self.errors

                keep_current = current_is_better.type(torch.uint8)
                keep_previous = (~current_is_better).type(torch.uint8)

                self.errors = (keep_previous * self.errors) + (keep_current * new_errors)
                self.root_rates = (keep_previous * self.root_rates) + (keep_current * new_root_rates)

    def _calc_rate_and_predictions_from_root(self, root_rates):
        positive = root_rates ** 2
        sum_per_row = positive.sum(axis=1).reshape(-1, 1)
        rates = positive / sum_per_row
        predictions = torch.matmul(rates, self.homogeneous)

        return rates, predictions

    def _get_optimizer(self, root_rates):
        optimizer = self.optimizer([root_rates], **self.optimizer_kwrgs)
        return optimizer

    def _validate_shapes(self):
        assert len(self.homogeneous.shape) == 2, 'homogeneous must be 2d tensor'
        assert len(self.observed.shape) == 2, 'observed must be 2d tensor'
        # assert self.homogeneous.shape[1] == self.observed.shape[1], 'must have the same features and number of features'

    def validate_initial_guess(self, initial_guess):
        assert len(initial_guess.shape) == 2, 'initial weights guess must be a 2d tensor'
        assert initial_guess.shape[0] == self.observed.shape[
            0], 'initial weights guess shape[0] must be same as observed shape[0]'
        assert initial_guess.shape[1] == self.homogeneous.shape[
            0], 'initial weights guess shape[1] must be same as homogeneous shape[0]'

    def _get_root_rates(self):
        root_rates = torch.full(
            (self.observed.shape[0], self.homogeneous.shape[0]),
            1,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True)
        return root_rates

    def _optimize(self, predictions, optimizer):
        if self.got_null:
            obs = torch.nan_to_num(self.observed, 0, posinf=0, neginf=0)
            pred = torch.nan_to_num(predictions, 0, posinf=0, neginf=0)
            output = self.loss(obs, pred)
        else:
            output = self.loss(self.observed, predictions)

        if type(output) == tuple:
            output, full_predictions = output
        else:
            full_predictions = None
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
        return full_predictions

    def _variables_to_cpu(self, rates):
        """
        1. Move rates to self.rates
        2. turns self.errors, self.rates, self.predictions to numpy
        3. deletes self.root_rates and rates
        :param rates:
        :return:
        """
        self.errors = self.errors.detach().cpu().numpy()
        self.predictions = self.predictions.detach().cpu().numpy()
        self.rates = rates.detach().cpu().numpy()

        del self.root_rates, rates
        self._clear_gpu_memory()

    def set_observed(self, observed):

        if observed is not None:
            self.observed = observed.detach()
            self.keep_values = None
            self.got_null = None
            self._clear_gpu_memory()

            got_null = observed.isnan().any().item()
            self.got_null = got_null
            if self.got_null:
                self.keep_values = \
                    (~observed[:,:self.homogeneous.shape[1]].isnan())\
                    .int().to(self.device)
            self.observed = observed.to(self.device)
        self._delete_all_errors()

    def unmix(self,
              iterations=250,
              initial_weights_guess: torch.Tensor = None,
              col_names = None):

        self._validate_shapes()
        if initial_weights_guess:
            self.validate_initial_guess(initial_weights_guess)
        root_rates = initial_weights_guess.to(self.device) \
            if initial_weights_guess \
            else self._get_root_rates()
        optimizer = self._get_optimizer(root_rates)

        for iter in range(iterations):
            rates, predictions = \
                self._calc_rate_and_predictions_from_root(root_rates)
            if iter != iterations - 1:
                full_predictions = self._optimize(predictions, optimizer)
            for_errors = predictions if full_predictions is None else full_predictions
            errors = torch.abs(self.observed - for_errors).sum(axis=1)\
                .reshape((-1, 1))
            self._keep_best_root_rates(errors, root_rates)

        # cleanup
        self.predictions = predictions if full_predictions is None else full_predictions
        self._variables_to_cpu(rates)

    def get_rates(self):
        return self.rates

    def get_predictions(self):
        return self.predictions

    def _recalc_reconstruction_error_2_side(self):
        """

        :return: row-wise difference between observed and predicted
        """
        errors = (self.observed.detach().cpu().numpy() - self.predictions)\
            .sum(axis=1).reshape((-1, 1))
        self.errors = errors

    def get_errors(self):
        self._recalc_reconstruction_error_2_side()
        return self.errors

    def get_cosine_similarity(self):
        predictions = torch.from_numpy(self.predictions).to(self.device)
        cosine_sim = (cosine_similarity(self.observed, predictions, 1))\
            .detach().cpu().numpy()
        del predictions
        self._clear_gpu_memory()
        return cosine_sim

    def get_errors_per_column(self):
        errors_per_column = self.observed.detach().cpu().numpy() - self.predictions
        return errors_per_column


if __name__ == '__main__':
    n_endmembers = 5
    n_features = 4
    n_rows = 1_000
    endmembers = torch.randint(1, 10, (n_endmembers, n_features), dtype=torch.float32)
    non_negative = torch.randn((n_rows, n_endmembers)) ** 2
    rates = non_negative / non_negative.sum(axis=1).reshape((-1, 1))
    observed = torch.matmul(rates, endmembers)
    lm = LinearUnmixing(endmembers, observed)
    lm.unmix()
    pred_rates = lm.get_rates()

    reconstruction_error_per_observation = lm.get_errors()
    print(f"first example - mean absolute reconstruction error: "
          f"{np.mean(np.abs(reconstruction_error_per_observation))}")
    detailed_reconstruction_error = lm.get_errors_per_column()
    print(f"first example - mean absolute reconstruction error per entry: "
          f"{np.mean(np.abs(detailed_reconstruction_error))}")
    rates_errors = (pred_rates - rates.numpy()).sum(axis=1).round(2)
    print(f"first example - max rate error: {rates_errors.max()}")


    with_nones = observed[:]
    with_nones[with_nones >= np.quantile(a = with_nones,
                                         q = 0.75)] = np.nan
    lm.set_observed(with_nones)
    lm.unmix()
    pred_rates = lm.get_rates()

    reconstruction_error_per_observation = lm.get_errors()
    print(f"second example - mean absolute reconstruction error with nans: "
          f"{np.nanmean(np.abs(reconstruction_error_per_observation))}")
    detailed_reconstruction_error = lm.get_errors_per_column()
    print(f"second example - mean absolute reconstruction error per entry with nans: "
          f"{np.nanmean(np.abs(detailed_reconstruction_error))}")
    rates_errors = (pred_rates - rates.numpy()).sum(axis=1).round(2)
    print(f"second example - max rate error with nans: {rates_errors.max()}")