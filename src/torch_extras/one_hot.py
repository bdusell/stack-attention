def indexes_to_one_hot_tensor(indexes, num_classes):
    result = torch.zeros(
        indexes.size() + (num_classes,),
        device=indexes.device
    )
    if result.size(1) > 0:
        result.scatter_(2, indexes.unsqueeze(2), 1)
    return result
