import io
import unittest
from contextlib import redirect_stderr

from torch.utils.data import DataLoader

from utils.regression_trainer import progress_dataloader


def _flatten_batches(progress):
    return [item for batch in progress for item in batch.tolist()]


class ProgressReportingTests(unittest.TestCase):
    def test_train_progress_iterates_over_complete_dataloader(self):
        dataloader = DataLoader(list(range(7)), batch_size=3, shuffle=False)
        output = io.StringIO()

        with redirect_stderr(output):
            observed = _flatten_batches(
                progress_dataloader(dataloader, 'Train epoch 3')
            )

        self.assertEqual(observed, list(range(7)))
        self.assertIn('Train epoch 3', output.getvalue())

    def test_evaluation_progress_iterates_over_complete_dataloader(self):
        dataloader = DataLoader(list(range(5)), batch_size=2, shuffle=False)
        output = io.StringIO()

        with redirect_stderr(output):
            observed = _flatten_batches(progress_dataloader(dataloader, 'Val'))

        self.assertEqual(observed, list(range(5)))
        self.assertIn('Val', output.getvalue())

    def test_progress_wrapper_preserves_dataset_and_sample_order(self):
        samples = [9, 2, 7, 1]
        dataloader = DataLoader(samples, batch_size=1, shuffle=False)

        with redirect_stderr(io.StringIO()):
            observed = _flatten_batches(
                progress_dataloader(dataloader, 'Test')
            )

        self.assertIs(dataloader.dataset, samples)
        self.assertEqual(observed, samples)


if __name__ == '__main__':
    unittest.main()
