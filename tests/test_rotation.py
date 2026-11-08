"""Bunge Euler angles and matrices agree with orix; misorientation sees the point group."""

import numpy as np
import pytest
import torch
from orix.quaternion import Rotation
from orix.quaternion.symmetry import Oh

from de_ebsd.core.rotation import (
    euler_to_matrix,
    euler_to_matrix_torch,
    matrix_to_euler,
    misorientation_deg,
)


@pytest.fixture(scope="module")
def rotations() -> Rotation:
    np.random.seed(11)
    return Rotation.random(64)


def test_matrices_are_orix(rotations):
    euler = rotations.to_euler()
    assert np.allclose(euler_to_matrix(euler), rotations.to_matrix(), atol=1e-12)


def test_matrices_keep_the_leading_shape(rotations):
    euler = rotations.to_euler().reshape(4, 16, 3)
    assert euler_to_matrix(euler).shape == (4, 16, 3, 3)
    assert euler_to_matrix(euler[0, 0]).shape == (3, 3)


def test_torch_matrices_are_the_numpy_ones_and_carry_gradients(rotations):
    euler = torch.tensor(rotations.to_euler(), requires_grad=True)
    matrices = euler_to_matrix_torch(euler)
    assert np.allclose(matrices.detach().numpy(), rotations.to_matrix(), atol=1e-12)
    matrices.sum().backward()
    assert euler.grad is not None and torch.isfinite(euler.grad).all()


def test_euler_round_trip(rotations):
    matrices = rotations.to_matrix()
    euler = matrix_to_euler(matrices)
    assert np.allclose(euler_to_matrix(euler), matrices, atol=1e-9)
    assert np.allclose(euler, rotations.to_euler(), atol=1e-9)


@pytest.mark.parametrize("Phi", [0.0, np.pi])
def test_euler_round_trip_at_the_gimbal_lock(Phi):
    matrix = euler_to_matrix([0.4, Phi, 0.3])
    euler = matrix_to_euler(matrix)
    assert euler[2] == 0.0
    assert np.allclose(euler_to_matrix(euler), matrix, atol=1e-12)


def test_misorientation_sees_through_the_point_group():
    np.random.seed(5)
    symmetry = Oh.to_matrix()
    g = Rotation.random(1).to_matrix()[0]
    turned = Rotation.from_axes_angles([0, 0, 1], 3, degrees=True).to_matrix()[0]
    equivalent = symmetry[7] @ g
    found = misorientation_deg(np.stack([equivalent, equivalent @ turned]), g, symmetry)
    assert np.allclose(found, [0.0, 3.0], atol=1e-4)
    # an orix Symmetry is accepted as it is
    assert np.allclose(misorientation_deg(equivalent[None], g, Oh), [0.0], atol=1e-4)
