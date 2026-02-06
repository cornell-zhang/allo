# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32

INPUT_DIMENSION = 13
POSSIBLE_OUTPUTS = 3
TRAINING_SETS = 163
NODES_PER_LAYER = 64
# LAYERS = 2
LEARNING_RATE = 0.01
# EPOCHS = 1
# TEST_SETS = 15
# NORM_PARAM = 0.05


def soft_max(
    net_outputs: float32[POSSIBLE_OUTPUTS], activations: float32[POSSIBLE_OUTPUTS]
):
    total: float32 = 0.0

    for i in range(POSSIBLE_OUTPUTS):
        total += allo.exp(-activations[i])
    for i in range(POSSIBLE_OUTPUTS):
        net_outputs[i] = allo.exp(-activations[i]) / total


def RELU_NPL(
    activations: float32[NODES_PER_LAYER], dactivations: float32[NODES_PER_LAYER]
):
    for i in range(NODES_PER_LAYER):
        dactivations[i] = activations[i] * (1.0 - activations[i])
        activations[i] = 1.0 / (1.0 + allo.exp(-activations[i]))


def RELU_PO(
    activations: float32[POSSIBLE_OUTPUTS], dactivations: float32[POSSIBLE_OUTPUTS]
):
    for i in range(POSSIBLE_OUTPUTS):
        dactivations[i] = activations[i] * (1.0 - activations[i])
        activations[i] = 1.0 / (1.0 + allo.exp(-activations[i]))


def add_bias_to_activations_NPL(
    biases: float32[NODES_PER_LAYER], activations: float32[NODES_PER_LAYER]
):
    for i in range(NODES_PER_LAYER):
        activations[i] = activations[i] + biases[i]


def add_bias_to_activations_PO(
    biases: float32[POSSIBLE_OUTPUTS], activations: float32[POSSIBLE_OUTPUTS]
):
    for i in range(POSSIBLE_OUTPUTS):
        activations[i] = activations[i] + biases[i]


def matrix_vector_product_with_bias_input_layer(
    biases: float32[NODES_PER_LAYER],
    weights: float32[INPUT_DIMENSION * NODES_PER_LAYER],
    activations: float32[NODES_PER_LAYER],
    input_sample: float32[INPUT_DIMENSION],
):

    for j in range(NODES_PER_LAYER):
        activations[j] = 0.0
        for i in range(INPUT_DIMENSION):
            activations[j] += weights[j * INPUT_DIMENSION + i] * input_sample[i]
    add_bias_to_activations_NPL(biases, activations)


def matrix_vector_product_with_bias_second_layer(
    biases: float32[NODES_PER_LAYER],
    weights: float32[NODES_PER_LAYER * NODES_PER_LAYER],
    activations: float32[NODES_PER_LAYER],
    input_activations: float32[NODES_PER_LAYER],
):

    for i in range(NODES_PER_LAYER):
        activations[i] = 0.0
        for j in range(NODES_PER_LAYER):
            activations[i] += weights[i * NODES_PER_LAYER + j] * input_activations[j]
    add_bias_to_activations_NPL(biases, activations)


def matrix_vector_product_with_bias_output_layer(
    biases: float32[POSSIBLE_OUTPUTS],
    weights: float32[NODES_PER_LAYER * POSSIBLE_OUTPUTS],
    activations: float32[POSSIBLE_OUTPUTS],
    input_activations: float32[NODES_PER_LAYER],
):

    for j in range(POSSIBLE_OUTPUTS):
        activations[j] = 0.0
        for i in range(NODES_PER_LAYER):
            activations[j] += weights[j * NODES_PER_LAYER + i] * input_activations[i]
    add_bias_to_activations_PO(biases, activations)


def take_difference(
    net_outputs: float32[POSSIBLE_OUTPUTS],
    solutions: float32[POSSIBLE_OUTPUTS],
    output_difference: float32[POSSIBLE_OUTPUTS],
    dactivations: float32[POSSIBLE_OUTPUTS],
):

    for i in range(POSSIBLE_OUTPUTS):
        output_difference[i] = (net_outputs[i] - solutions[i]) * -1.0 * dactivations[i]


def get_delta_matrix_weights3(
    delta_weights3: float32[NODES_PER_LAYER * POSSIBLE_OUTPUTS],
    output_difference: float32[POSSIBLE_OUTPUTS],
    last_activations: float32[NODES_PER_LAYER],
):

    for i, j in allo.grid(NODES_PER_LAYER, POSSIBLE_OUTPUTS):
        delta_weights3[i * POSSIBLE_OUTPUTS + j] = (
            last_activations[i] * output_difference[j]
        )


def get_oracle_activations2(
    weights3: float32[NODES_PER_LAYER * POSSIBLE_OUTPUTS],
    output_differences: float32[POSSIBLE_OUTPUTS],
    oracle_activations: float32[NODES_PER_LAYER],
    dactivations: float32[NODES_PER_LAYER],
):

    for i in range(NODES_PER_LAYER):
        oracle_activations[i] = 0.0
        for j in range(POSSIBLE_OUTPUTS):
            oracle_activations[i] += (
                output_differences[j] * weights3[i * POSSIBLE_OUTPUTS + j]
            )
        oracle_activations[i] = oracle_activations[i] * dactivations[i]


def get_delta_matrix_weights2(
    delta_weights2: float32[NODES_PER_LAYER * NODES_PER_LAYER],
    output_difference: float32[NODES_PER_LAYER],
    last_activations: float32[NODES_PER_LAYER],
):

    for i, j in allo.grid(NODES_PER_LAYER, NODES_PER_LAYER):
        delta_weights2[i * NODES_PER_LAYER + j] = (
            last_activations[i] * output_difference[j]
        )


def get_oracle_activations1(
    weights2: float32[NODES_PER_LAYER * NODES_PER_LAYER],
    output_differences: float32[NODES_PER_LAYER],
    oracle_activations: float32[NODES_PER_LAYER],
    dactivations: float32[NODES_PER_LAYER],
):

    for i in range(NODES_PER_LAYER):
        oracle_activations[i] = 0.0
        for j in range(NODES_PER_LAYER):
            oracle_activations[i] += (
                output_differences[j] * weights2[i * NODES_PER_LAYER + j]
            )
        oracle_activations[i] = oracle_activations[i] * dactivations[i]


def get_delta_matrix_weights1(
    delta_weights1: float32[INPUT_DIMENSION * NODES_PER_LAYER],
    output_difference: float32[NODES_PER_LAYER],
    last_activations: float32[INPUT_DIMENSION],
):

    for i, j in allo.grid(INPUT_DIMENSION, NODES_PER_LAYER):
        delta_weights1[i * NODES_PER_LAYER + j] = (
            last_activations[i] * output_difference[j]
        )


def update_weights(
    weights1: float32[INPUT_DIMENSION * NODES_PER_LAYER],
    weights2: float32[NODES_PER_LAYER * NODES_PER_LAYER],
    weights3: float32[NODES_PER_LAYER * POSSIBLE_OUTPUTS],
    d_weights1: float32[INPUT_DIMENSION * NODES_PER_LAYER],
    d_weights2: float32[NODES_PER_LAYER * NODES_PER_LAYER],
    d_weights3: float32[NODES_PER_LAYER * POSSIBLE_OUTPUTS],
    biases1: float32[NODES_PER_LAYER],
    biases2: float32[NODES_PER_LAYER],
    biases3: float32[POSSIBLE_OUTPUTS],
    d_biases1: float32[NODES_PER_LAYER],
    d_biases2: float32[NODES_PER_LAYER],
    d_biases3: float32[POSSIBLE_OUTPUTS],
):

    # TODO: allo.sqrt() does not currently work with float32
    norm1: float32 = 0.0
    bias_norm1: float32 = 0.0

    for i, j in allo.grid(INPUT_DIMENSION, NODES_PER_LAYER):
        weights1[i * NODES_PER_LAYER + j] -= (
            d_weights1[i * NODES_PER_LAYER + j] * LEARNING_RATE
        )
        norm1 += weights1[i * NODES_PER_LAYER + j] * weights1[i * NODES_PER_LAYER + j]
    for i in range(NODES_PER_LAYER):
        biases1[i] -= d_biases1[i] * LEARNING_RATE
        bias_norm1 += biases1[i] * biases1[i]

    norm1_sqrt = allo.sqrt(norm1)
    bias_norm1_sqrt = allo.sqrt(bias_norm1)

    for i, j in allo.grid(INPUT_DIMENSION, NODES_PER_LAYER):
        weights1[i * NODES_PER_LAYER + j] = (
            weights1[i * NODES_PER_LAYER + j] / norm1_sqrt
        )
    for i in range(NODES_PER_LAYER):
        biases1[i] = biases1[i] / bias_norm1_sqrt

    norm2: float32 = 0.0
    bias_norm2: float32 = 0.0

    for i, j in allo.grid(NODES_PER_LAYER, NODES_PER_LAYER):
        weights2[i * NODES_PER_LAYER + j] -= (
            d_weights2[i * NODES_PER_LAYER + j] * LEARNING_RATE
        )
        norm2 += weights2[i * NODES_PER_LAYER + j] * weights2[i * NODES_PER_LAYER + j]
    for i in range(NODES_PER_LAYER):
        biases2[i] -= d_biases2[i] * LEARNING_RATE
        bias_norm2 += biases2[i] * biases2[i]

    norm2_sqrt = allo.sqrt(norm2)
    bias_norm2_sqrt = allo.sqrt(bias_norm2)

    for i, j in allo.grid(NODES_PER_LAYER, NODES_PER_LAYER):
        weights2[i * NODES_PER_LAYER + j] = (
            weights2[i * NODES_PER_LAYER + j] / norm2_sqrt
        )
    for i in range(NODES_PER_LAYER):
        biases2[i] = biases2[i] / bias_norm2_sqrt

    norm3: float32 = 0.0
    bias_norm3: float32 = 0.0

    for i, j in allo.grid(NODES_PER_LAYER, POSSIBLE_OUTPUTS):
        weights3[i * POSSIBLE_OUTPUTS + j] -= (
            d_weights3[i * POSSIBLE_OUTPUTS + j] * LEARNING_RATE
        )
        norm3 += weights3[i * POSSIBLE_OUTPUTS + j] * weights3[i * POSSIBLE_OUTPUTS + j]
    for i in range(POSSIBLE_OUTPUTS):
        biases3[i] -= d_biases3[i] * LEARNING_RATE
        bias_norm3 += biases3[i] * biases3[i]

    norm3_sqrt = allo.sqrt(norm3)
    bias_norm3_sqrt = allo.sqrt(bias_norm3)

    for i, j in allo.grid(NODES_PER_LAYER, POSSIBLE_OUTPUTS):
        weights3[i * POSSIBLE_OUTPUTS + j] = (
            weights3[i * POSSIBLE_OUTPUTS + j] / norm3_sqrt
        )
    for i in range(POSSIBLE_OUTPUTS):
        biases3[i] = biases3[i] / bias_norm3_sqrt


def backprop(
    weights1: float32[INPUT_DIMENSION * NODES_PER_LAYER],
    weights2: float32[NODES_PER_LAYER * NODES_PER_LAYER],
    weights3: float32[NODES_PER_LAYER * POSSIBLE_OUTPUTS],
    biases1: float32[NODES_PER_LAYER],
    biases2: float32[NODES_PER_LAYER],
    biases3: float32[POSSIBLE_OUTPUTS],
    training_data: float32[TRAINING_SETS * INPUT_DIMENSION],
    training_targets: float32[TRAINING_SETS * POSSIBLE_OUTPUTS],
):

    # Forward and training structures
    activations1: float32[NODES_PER_LAYER] = 0.0
    activations2: float32[NODES_PER_LAYER] = 0.0
    activations3: float32[POSSIBLE_OUTPUTS] = 0.0
    dactivations1: float32[NODES_PER_LAYER] = 0.0
    dactivations2: float32[NODES_PER_LAYER] = 0.0
    dactivations3: float32[POSSIBLE_OUTPUTS] = 0.0
    net_outputs: float32[POSSIBLE_OUTPUTS] = 0.0
    # Training structures
    output_difference: float32[POSSIBLE_OUTPUTS] = 0.0
    delta_weights1: float32[INPUT_DIMENSION * NODES_PER_LAYER] = 0.0
    delta_weights2: float32[NODES_PER_LAYER * NODES_PER_LAYER] = 0.0
    delta_weights3: float32[NODES_PER_LAYER * POSSIBLE_OUTPUTS] = 0.0
    oracle_activations1: float32[NODES_PER_LAYER] = 0.0
    oracle_activations2: float32[NODES_PER_LAYER] = 0.0

    for i in range(TRAINING_SETS):
        for j in range(NODES_PER_LAYER):
            activations1[j] = 0.0
            activations2[j] = 0.0
            if j < POSSIBLE_OUTPUTS:
                activations3[j] = 0.0

        training_data_input1: float32[INPUT_DIMENSION] = 0.0
        for k in range(INPUT_DIMENSION):
            training_data_input1[k] = training_data[i * INPUT_DIMENSION + k]
        matrix_vector_product_with_bias_input_layer(
            biases1, weights1, activations1, training_data_input1
        )

        RELU_NPL(activations1, dactivations1)
        matrix_vector_product_with_bias_second_layer(
            biases2, weights2, activations2, activations1
        )
        RELU_NPL(activations2, dactivations2)
        matrix_vector_product_with_bias_output_layer(
            biases3, weights3, activations3, activations2
        )
        RELU_PO(activations3, dactivations3)
        soft_max(net_outputs, activations3)

        training_targets_input: float32[POSSIBLE_OUTPUTS] = 0.0
        for k in range(POSSIBLE_OUTPUTS):
            training_targets_input[k] = training_targets[i * POSSIBLE_OUTPUTS + k]
        take_difference(
            net_outputs, training_targets_input, output_difference, dactivations3
        )

        get_delta_matrix_weights3(delta_weights3, output_difference, activations2)
        get_oracle_activations2(
            weights3, output_difference, oracle_activations2, dactivations2
        )
        get_delta_matrix_weights2(delta_weights2, oracle_activations2, activations1)
        get_oracle_activations1(
            weights2, oracle_activations2, oracle_activations1, dactivations1
        )

        training_data_input2: float32[INPUT_DIMENSION] = 0.0
        for k in range(INPUT_DIMENSION):
            training_data_input2[k] = training_data[i * INPUT_DIMENSION + k]
        get_delta_matrix_weights1(
            delta_weights1, oracle_activations1, training_data_input2
        )

        update_weights(
            weights1,
            weights2,
            weights3,
            delta_weights1,
            delta_weights2,
            delta_weights3,
            biases1,
            biases2,
            biases3,
            oracle_activations1,
            oracle_activations2,
            output_difference,
        )
