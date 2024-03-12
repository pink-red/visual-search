from timm.layers import ClassifierHead
import torch


"""
Вычислять вероятности всех 10861 тегов, как происходит по умолчанию, затратно
по времени и по памяти. Альтернатива: вычисляем и храним только эмбеддинги, а
вероятности тегов вычисляем только когда это необходимо.

Для этого, извлекаем из head-части модели, в которой классификаторы для всех
тегов, по одному классификатору для каждого нужного тега и прогоняем эмбеддинги
через них.
"""


def extract_single_classifier(model, classifier_idx: int):
    # Создаем head-часть модели на 1 тег
    small_head = ClassifierHead(
        in_features=model.num_features,
        num_classes=1,
        pool_type="avg",
        drop_rate=0,
        input_fmt=model.output_fmt,
    )

    # Отключаем pooling, эмбеддинги уже прошли через этот слой в
    # ModelV3.images_to_embeddings().
    small_head.global_pool = torch.nn.Identity()

    # Копируем в однотеговую модель веса нужного классификатора
    orig_state_dict = model.head.fc.state_dict()
    small_state_dict = {
        "weight": orig_state_dict["weight"][[classifier_idx], :],
        "bias": orig_state_dict["bias"][[classifier_idx]],
    }
    small_head.fc.load_state_dict(small_state_dict)

    return small_head
