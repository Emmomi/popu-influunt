import tensorflow as tf

class pif_Model(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)