from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from finalization.artifacts import RunRecord
from finalization.score_approximation import CellSpec, config_fingerprint
from finalization.score_local_quadrature import (
    _summary_rows,
    autograd_local_box_score,
    cell_record_path,
    compute_local_score_metrics,
    filter_cell_specs,
    fisher_gauss_newton_scales,
    gauss_legendre_tensor_rule,
    load_local_quadrature_config,
    local_box_quadrature_score,
    pending_cell_specs,
    resolve_quadrature_epsilon_dim,
)
from models.vi_model import ConditionalGaussian


class LinearGaussianNet(torch.nn.Module):

    def __init__(
        self,
        matrix: torch.Tensor,
        variance: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)
        raw_variance = torch.log(torch.expm1(variance))
        self.register_buffer("raw_variance", raw_variance)

    def forward(self, epsilon: torch.Tensor) -> torch.Tensor:
        mu = epsilon @ self.matrix.transpose(0, 1)
        raw = self.raw_variance.expand(
            epsilon.shape[0],
            -1,
        )
        return torch.cat([mu, raw], dim=-1)


class LinearConditionalGaussian(torch.nn.Module):

    def __init__(
        self,
        matrix: torch.Tensor,
        variance: torch.Tensor,
    ) -> None:
        super().__init__()
        self.net = LinearGaussianNet(matrix, variance)
        self.epsilon_dim = matrix.shape[-1]
        self.z_dim = matrix.shape[0]

    def _variance_from_raw(
        self,
        raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        variance = torch.nn.functional.softplus(raw)
        return variance, variance.log()


def make_model() -> ConditionalGaussian:
    cfg = OmegaConf.create({
        "z_dim": 2,
        "epsilon_dim": 4,
        "hidden_dim": 8,
        "num_layers": 1,
        "device": "cpu",
        "uniform": False,
    })
    return ConditionalGaussian(cfg).to(dtype=torch.float64)


class LocalQuadratureTest(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_default_config_locks_requested_rule(self) -> None:
        cfg = load_local_quadrature_config(None)
        self.assertEqual(int(cfg.evaluation.quadrature.order), 13)
        self.assertEqual(
            float(cfg.evaluation.quadrature.standardized_half_width),
            4.0,
        )
        self.assertEqual(list(cfg.selection.seeds), [42, 43, 44, 45, 46])
        self.assertEqual(
            str(cfg.evaluation.quadrature.epsilon_dim),
            "auto_from_checkpoint",
        )

    def test_quadrature_dimension_is_checkpoint_derived(self) -> None:
        model = make_model()
        self.assertEqual(
            resolve_quadrature_epsilon_dim(
                model,
                "auto_from_checkpoint",
            ),
            4,
        )
        with self.assertRaisesRegex(
            ValueError,
            "does not match checkpoint dimension",
        ):
            resolve_quadrature_epsilon_dim(model, 2)

    def test_tensor_rule_has_expected_mass_and_node_count(self) -> None:
        nodes, log_weights = gauss_legendre_tensor_rule(
            dimension=4,
            order=13,
            half_width=4.0,
        )
        self.assertEqual(tuple(nodes.shape), (13**4, 4))
        self.assertEqual(tuple(log_weights.shape), (13**4,))
        self.assertAlmostEqual(
            float(torch.exp(log_weights).sum().item()),
            8.0**4,
            places=9,
        )

    def test_fisher_scale_matches_linear_gaussian_formula(self) -> None:
        matrix = torch.tensor(
            [[1.0, 2.0], [-0.5, 0.25]],
            dtype=torch.float64,
        )
        variance = torch.tensor([0.5, 2.0], dtype=torch.float64)
        model = LinearConditionalGaussian(matrix, variance)
        epsilon = torch.randn(3, 2, dtype=torch.float64)
        transform, log_det, diagnostics = fisher_gauss_newton_scales(
            model,
            epsilon,
            batch_size=2,
            max_eigenvalue=1.0e6,
        )
        expected_fisher = (
            torch.eye(2, dtype=torch.float64)
            + matrix.transpose(0, 1)
            @ torch.diag(variance.reciprocal())
            @ matrix
        )
        values, vectors = torch.linalg.eigh(expected_fisher)
        expected_transform = (
            vectors
            @ torch.diag(values.rsqrt())
            @ vectors.transpose(0, 1)
        )
        torch.testing.assert_close(
            transform,
            expected_transform.expand_as(transform),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        torch.testing.assert_close(
            log_det,
            torch.full_like(
                log_det,
                -0.5 * torch.logdet(expected_fisher),
            ),
        )
        self.assertGreaterEqual(diagnostics["fisher_eigenvalue_min"], 1.0)

    def test_streamed_analytic_score_matches_autograd(self) -> None:
        model = make_model()
        epsilon = torch.randn(3, 4, dtype=torch.float64)
        z, _ = model.forward(epsilon)
        transform, log_det, _ = fisher_gauss_newton_scales(
            model,
            epsilon,
            batch_size=3,
            max_eigenvalue=1.0e6,
        )
        nodes, log_weights = gauss_legendre_tensor_rule(
            dimension=4,
            order=3,
            half_width=2.0,
        )
        actual, _, diagnostics = local_box_quadrature_score(
            model,
            z,
            epsilon,
            transform,
            log_det,
            nodes=nodes,
            log_weights=log_weights,
            boundary_inner_half_width=1.5,
            z_chunk_size=2,
            node_chunk_size=17,
        )
        expected = autograd_local_box_score(
            model,
            z,
            epsilon,
            transform,
            log_det,
            nodes=nodes,
            log_weights=log_weights,
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        self.assertEqual(
            diagnostics["quadrature_conditional_evaluations"],
            3 * 3**4,
        )

    def test_order_13_resolves_sharp_linear_reverse(self) -> None:
        matrix = torch.ones(1, 1, dtype=torch.float64)
        variance = torch.tensor([0.01], dtype=torch.float64)
        model = LinearConditionalGaussian(matrix, variance)
        z = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
        posterior_mean = z / 1.01
        transform, log_det, _ = fisher_gauss_newton_scales(
            model,
            posterior_mean,
            batch_size=3,
            max_eigenvalue=1.0e6,
        )
        nodes, log_weights = gauss_legendre_tensor_rule(
            dimension=1,
            order=13,
            half_width=4.0,
        )
        score, log_q, diagnostics = local_box_quadrature_score(
            model,
            z,
            posterior_mean,
            transform,
            log_det,
            nodes=nodes,
            log_weights=log_weights,
            boundary_inner_half_width=3.5,
            z_chunk_size=3,
            node_chunk_size=13,
        )
        marginal_variance = 1.01
        expected_score = -z / marginal_variance
        expected_log_q = -0.5 * (
            math.log(2.0 * math.pi * marginal_variance)
            + z.square().squeeze(-1) / marginal_variance
        )
        torch.testing.assert_close(
            score,
            expected_score,
            rtol=0.0,
            atol=2.0e-4,
        )
        torch.testing.assert_close(
            log_q,
            expected_log_q,
            rtol=0.0,
            atol=2.0e-4,
        )
        self.assertLess(
            diagnostics["quadrature_boundary_mass_max"],
            1.0e-3,
        )

    def test_order_13_conditional_gaussian_cpu_smoke(self) -> None:
        model = make_model()
        epsilon = torch.randn(4, 4, dtype=torch.float64)
        z, _ = model.forward(epsilon)
        transform, log_det, _ = fisher_gauss_newton_scales(
            model,
            epsilon,
            batch_size=2,
            max_eigenvalue=1.0e6,
        )
        nodes, log_weights = gauss_legendre_tensor_rule(
            dimension=4,
            order=13,
            half_width=4.0,
        )
        score, log_q, diagnostics = local_box_quadrature_score(
            model,
            z,
            epsilon,
            transform,
            log_det,
            nodes=nodes,
            log_weights=log_weights,
            boundary_inner_half_width=3.5,
            z_chunk_size=2,
            node_chunk_size=4096,
        )
        self.assertEqual(tuple(score.shape), (4, 2))
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(torch.isfinite(log_q).all())
        self.assertEqual(
            diagnostics["quadrature_nodes_per_z"],
            13**4,
        )

    def test_score_metric_definitions(self) -> None:
        method = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        local = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        target = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        metrics = compute_local_score_metrics(method, local, target)
        self.assertAlmostEqual(float(metrics["method_l2"]), 1.0)
        self.assertAlmostEqual(float(metrics["local_target_l2"]), 2.0)
        self.assertAlmostEqual(float(metrics["method_target_l2"]), 1.0)

    def test_runtime_seed_filter_does_not_change_fingerprint(self) -> None:
        cfg = load_local_quadrature_config(None)
        fingerprint = config_fingerprint(cfg)
        records = [
            RunRecord(
                run_id=f"run-{seed}",
                seed=seed,
                method="SIVI",
                target="x_shaped",
                runner_type="SIVI",
                config_path=Path("config.yaml"),
                result_path=Path("results"),
                duration_sec=None,
                status="completed",
                entry={},
            )
            for seed in (42, 44)
        ]
        specs = [
            CellSpec(
                record=record,
                progress=1.0,
                epoch=10000,
                checkpoint_dir=Path("checkpoint"),
            )
            for record in records
        ]
        selected = filter_cell_specs(specs, seeds=[44])
        self.assertEqual([spec.record.seed for spec in selected], [44])
        self.assertEqual(config_fingerprint(cfg), fingerprint)

    def test_one_seed_summary_has_one_row_per_cell_group(self) -> None:
        diagnostics = {
            "quadrature_nodes_per_second": 10.0,
            "quadrature_ess_mean": 20.0,
            "quadrature_max_weight_p95": 0.1,
            "quadrature_boundary_mass_p95": 0.01,
            "physical_scale_min": 0.2,
            "physical_scale_median": 0.5,
            "physical_scale_max": 1.0,
        }
        records = [
            {
                "target": "x_shaped",
                "method": method,
                "progress": 1.0,
                "epoch": 10000,
                "quadrature_epsilon_dim": 2,
                "method_l2": 1.0,
                "method_relative_l2": 0.5,
                "method_target_l2": 2.0,
                "local_target_l2": 3.0,
                "method_runtime_sec": 0.1,
                "scaling_runtime_sec": 0.2,
                "quadrature_runtime_sec": 0.3,
                "total_runtime_sec": 0.6,
                "peak_gpu_reserved_bytes": 1024,
                "gpu_headroom_gib": 20.0,
                "diagnostics": diagnostics,
            }
            for method in ("SIVI", "UIVI")
        ]
        rows = _summary_rows(records)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["n_seeds"] == 1 for row in rows))
        self.assertTrue(
            all(row["method_l2_sd"] is None for row in rows)
        )

    def test_resume_reuses_runtime_filtered_cell(self) -> None:
        record = RunRecord(
            run_id="run-44",
            seed=44,
            method="SIVI",
            target="x_shaped",
            runner_type="SIVI",
            config_path=Path("config.yaml"),
            result_path=Path("results"),
            duration_sec=None,
            status="completed",
            entry={},
        )
        spec = CellSpec(
            record=record,
            progress=1.0,
            epoch=10000,
            checkpoint_dir=Path("checkpoint"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fingerprint = "fixed-estimator"
            path = cell_record_path(root, spec)
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps({
                    "analysis_fingerprint": fingerprint,
                    "cell_key": spec.key,
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                pending_cell_specs(
                    [spec],
                    run_root=root,
                    fingerprint=fingerprint,
                    resume=True,
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
