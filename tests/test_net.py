import pytest
import torch
import numpy as np

from typing import Callable, Tuple
from torch import Tensor
from torch.nn import Module
from nn import FeedForwardNetwork, DuelingQNetwork


class BaseNetworkTest:
    @pytest.fixture(scope="session")
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture(params=[1, 2, np.random.randint(3, 32)])
    def batch_size(self, request) -> int:
        return request.param

    @pytest.fixture
    def state_space(self) -> int:
        return 4

    @pytest.fixture
    def action_space(self) -> int:
        return 8

    @pytest.fixture
    def build_model(self, *args, **kwargs) -> Callable[..., Module]:
        raise NotImplementedError

    @pytest.fixture
    def model(self, build_model, dims, device) -> Module:
        return build_model(*dims, device=device)

    @staticmethod
    def assert_model_output(x: Tensor, batch_size: int, action_space: int, is_batched: bool = True):
        assert isinstance(x, Tensor)
        assert torch.isfinite(x).all()
        if is_batched:
            assert x.shape == (batch_size, action_space)
        else:
            assert x.shape == (1, action_space)

    @staticmethod
    def test_init_invalid_dims(dims_invalid, build_model, device):
        with pytest.raises(ValueError):
            build_model(*dims_invalid, device=device)

    @staticmethod
    def test_init_valid_dims(dims, build_model, device):
        net = build_model(*dims, device=device)
        assert isinstance(net, Module)

    def test_forward_batched(self, model, state_space, action_space, batch_size, device):
        x = torch.randn(batch_size, state_space).to(device=device)
        out = model.forward(x)
        self.assert_model_output(out, batch_size=batch_size, action_space=action_space, is_batched=True)

    def test_forward_single(self, model, state_space, action_space, device):
        x = torch.randn(1, state_space).to(device=device)
        out = model.forward(x)
        self.assert_model_output(out, batch_size=1, action_space=action_space, is_batched=False)

    @staticmethod
    def test_last_layer_is_identity(model, device):
        last_module = list(model.modules())[-1]
        assert isinstance(last_module, torch.nn.Identity)


class TestFeedForwardNetwork(BaseNetworkTest):
    @pytest.fixture(
        params=[(4, 8), (4, 1, 8), (4, 128, 128, 8)],
        ids=["no_hidden_dimension", "minimal", "normal"]
    )
    def dims(self, request) -> Tuple[int, ...]:
        """Set of valid model dimensions"""
        return request.param

    @pytest.fixture(
        params=[(0,), (1,), (0, 1, 8), (4, 1, 0)],
        ids=["zero_dimension", "single_dimension", "zero_dimension_in", "zero_dimension_out"]
    )
    def dims_invalid(self, request) -> Tuple[int, ...]:
        """Set of invalid model dimensions"""
        return request.param

    @pytest.fixture
    def build_model(self) -> Callable[..., FeedForwardNetwork]:
        def _model(*args, **kwargs):
            return FeedForwardNetwork(*args, **kwargs)
        return _model


class TestDuelingQNetwork(BaseNetworkTest):
    @pytest.fixture(
        params=[(4, 1, 8), (4, 128, 128, 8)],
        ids=["minimal", "normal"]
    )
    def dims(self, request) -> Tuple[int, ...]:
        """Set of valid model dimensions"""
        return request.param

    @pytest.fixture(
        params=[(0,), (1,), (4, 8), (0, 1, 8), (4, 1, 0)],
        ids=["zero_dimension", "single_dimension", "no_hidden_dimension", "zero_dimension_in", "zero_dimension_out"]
    )
    def dims_invalid(self, request) -> Tuple[int, ...]:
        """Set of invalid model dimensions"""
        return request.param

    @pytest.fixture(
        params=[1, np.random.randint(2, 8)],
        ids=["single", "more"]
    )
    def encoder_depth(self, request) -> int:
        return request.param

    @pytest.fixture(
        params=[-1, 0],
        ids=["negative_depth", "no_depth"]
    )
    def encoder_depth_invalid(self, request) -> int:
        return request.param

    @staticmethod
    def test_init_valid_encoder_depth(dims, encoder_depth, build_model, device):
        net = build_model(*dims, encoder_depth=encoder_depth, device=device)
        assert isinstance(net, DuelingQNetwork)

    @staticmethod
    def test_init_invalid_encoder_depth(dims, encoder_depth_invalid, build_model, device):
        with pytest.raises(ValueError):
            build_model(*dims, encoder_depth=encoder_depth_invalid, device=device)

    @pytest.fixture(scope="class")
    def build_model(self) -> Callable[..., DuelingQNetwork]:
        def _make(*args, **kwargs):
            return DuelingQNetwork(*args, **kwargs)
        return _make
