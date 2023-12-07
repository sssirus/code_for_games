from dataclasses import dataclass

from torchtyping import TensorType


@dataclass
class DPORLElement:
    """
    :param query_tensor: The query tensor i.e. the prompt tokens.
                         Should be a long tensor.
    :type query_tensor: torch.Tensor

    :param response_tensor: The response tensor i.e. the output tokens.
                            Should be a long tensor.
    :type response_tensor: torch.Tensor

    :param logprobs: The log probabilities over the response tokens generated
                    by the policy network (i.e. the autoregressive model).
                    Should be a float tensor of same size as tokens.
    :type logprobs: torch.Tensor

    :param values: The values for each token generated from the value network or value head.
                    Should be a float tensor of same size as tokens.
    :type values: torch.Tensor

    :param rewards: The rewards for each token outputted in response.
                    Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    chosen_input_ids: TensorType["batch_size", "chosen_size"]
    chosen_attention_masks: TensorType["batch_size", "chosen_size"]
    chosen_labels: TensorType["batch_size", "chosen_size"]
    rejected_input_ids: TensorType["batch_size", "rejected_size"]
    rejected_attention_masks: TensorType["batch_size", "rejected_size"]
    rejected_labels: TensorType["batch_size", "rejected_size"]


@dataclass
class DPORLBatch:
    """
    A batched version of the PPORLElement. See PPORLElement for more details on individual fields.

    :param query_tensors: A batch of query tensors. Should be a long tensor.
    :type query_tensors: torch.Tensor

    :param response_tensors: A batch of response tensors. Should be a long tensor.
    :type response_tensors: torch.Tensor

    :param logprobs: A batch of log probabilities from policy
    :type logprobs: torch.Tensor

    :param values: A batch of values from value network
    :type values: torch.Tensor

    :param rewards: A batch of rewards
    :type rewards: torch.Tensor
    """

    chosen_input_ids: TensorType["batch_size", "chosen_size"]
    chosen_attention_masks: TensorType["batch_size", "chosen_size"]
    chosen_labels: TensorType["batch_size", "chosen_size"]
    rejected_input_ids: TensorType["batch_size", "rejected_size"]
    rejected_attention_masks: TensorType["batch_size", "rejected_size"]
    rejected_labels: TensorType["batch_size", "rejected_size"]
