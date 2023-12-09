from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    concatenate,
    LSTM,
    GRU,
    BatchNormalization,
    Flatten,
    Input,
    RepeatVector,
    Permute,
    multiply,
    Lambda,
    Activation,
)
from tensorflow.keras.optimizers import Adam


def birnn_model(
        input_shape, num_output, num_rnn_unit=128, num_dense_unit=250, rnn_type="gru"
):
    if rnn_type not in ["lstm", "gru"]:
        print("Wrong RNN type.")
        return

    input_1 = Input(shape=input_shape, dtype="float32")
    if rnn_type == "lstm":
        forwards_1 = LSTM(num_rnn_unit, return_sequences=True, dropout=0.2)(input_1)
    else:
        forwards_1 = GRU(num_rnn_unit, return_sequences=True, dropout=0.2)(input_1)
    if rnn_type == "lstm":
        backwards_1 = LSTM(
            num_rnn_unit, return_sequences=True, dropout=0.2, go_backwards=True
        )(input_1)
    else:
        backwards_1 = GRU(
            num_rnn_unit, return_sequences=True, dropout=0.2, go_backwards=True
        )(input_1)
    merged = concatenate([forwards_1, backwards_1])
    after_merge = Dense(num_dense_unit, activation="relu")(merged)
    after_dp = Dropout(0.4)(after_merge)
    output = Dense(num_output, activation="softmax")(after_dp)
    model = Model(inputs=input_1, outputs=output)
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"]
    )

    # model.summary()

    return model
