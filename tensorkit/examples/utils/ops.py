from tensorkit import tensor as T

__all__ = ['calculate_acc']


def calculate_acc(logits: T.Tensor, y: T.Tensor) -> T.Tensor:
    with T.no_grad():
        out_y = T.argmax(logits, axis=-1)
        return T.reduce_mean(T.cast(T.equal(out_y, y), dtype=T.float32))
