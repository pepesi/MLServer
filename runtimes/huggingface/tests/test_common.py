import pytest
import json

import numpy as np
import uuid
from transformers.pipelines import Conversation
from mlserver_huggingface.common import CommonJSONEncoder


@pytest.mark.parametrize(
    "output, expected",
    [
        (
            {
                "f_": np.float_(1),
                "f16": np.float_(1),
                "f32": np.float_(1),
                "f64": np.float_(1),
                "i_": np.int_(1.0),
                "i0": np.int0(1.0),
                "i8": np.int8(1.0),
                "i16": np.int16(1.0),
                "i32": np.int32(1.0),
                "i64": np.int64(1.0),
            },
            '{"f_": 1.0, "f16": 1.0, "f32": 1.0, "f64": 1.0, "i_": '
            + '1, "i0": 1, "i8": 1, "i16": 1, "i32": 1, "i64": 1}',
        ),
        (
            {
                "ints": np.full(1, 1, dtype=np.int8),
                "floats": np.full(1, 1, dtype=np.float32),
            },
            '{"ints": [1], "floats": [1.0]}',
        ),
        (np.full((2, 2, 2), 1, dtype=np.int8), "[[[1, 1], [1, 1]], [[1, 1], [1, 1]]]"),
        (
            Conversation(
                text="hello",
                conversation_id=uuid.UUID('712dcbad-a042-4d9d-ab4d-84f20d6d9e7e'),
                generated_responses=["hello!"],
            ),
            '{"uuid": "712dcbad-a042-4d9d-ab4d-84f20d6d9e7e", "past_user_inputs": [],'
            + ' "generated_responses": ["hello!"], "new_user_input": "hello"}',
        ),
    ],
)
def test_json_encoder(output, expected):
    assert json.dumps(output, cls=CommonJSONEncoder) == expected
