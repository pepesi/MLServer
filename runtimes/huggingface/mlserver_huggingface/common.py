import os
import io
import base64
import json
import uuid
from typing import Optional, Dict
from distutils.util import strtobool
from PIL import Image

import numpy as np
from pydantic import BaseSettings
from mlserver.errors import MLServerError

from transformers.pipelines import pipeline, Conversation
from transformers.pipelines.base import Pipeline
from transformers.models.auto.tokenization_auto import AutoTokenizer

from optimum.pipelines import SUPPORTED_TASKS as SUPPORTED_OPTIMUM_TASKS


HUGGINGFACE_TASK_TAG = "task"

ENV_PREFIX_HUGGINGFACE_SETTINGS = "MLSERVER_MODEL_HUGGINGFACE_"
HUGGINGFACE_PARAMETERS_TAG = "huggingface_parameters"
PARAMETERS_ENV_NAME = "PREDICTIVE_UNIT_PARAMETERS"

CONTENT_TYPE_IMAGE = "image"
CONTENT_TYPE_JSON = "json"
CONTENT_TYPE_CONVERSATION = "conversation"


class InvalidTranformerInitialisation(MLServerError):
    def __init__(self, code: int, reason: str):
        super().__init__(
            f"Huggingface server failed with {code}, {reason}",
            status_code=code,
        )


class HuggingFaceSettings(BaseSettings):
    """
    Parameters that apply only to alibi huggingface models
    """

    class Config:
        env_prefix = ENV_PREFIX_HUGGINGFACE_SETTINGS

    task: str = ""
    pretrained_model: Optional[str] = None
    pretrained_tokenizer: Optional[str] = None
    optimum_model: bool = False
    device: int = -1
    batch_size: Optional[int] = None


def parse_parameters_from_env() -> Dict:
    """
    TODO
    """
    parameters = json.loads(os.environ.get(PARAMETERS_ENV_NAME, "[]"))

    type_dict = {
        "INT": int,
        "FLOAT": float,
        "DOUBLE": float,
        "STRING": str,
        "BOOL": bool,
    }

    parsed_parameters = {}
    for param in parameters:
        name = param.get("name")
        value = param.get("value")
        type_ = param.get("type")
        if type_ == "BOOL":
            parsed_parameters[name] = bool(strtobool(value))
        else:
            try:
                parsed_parameters[name] = type_dict[type_](value)
            except ValueError:
                raise InvalidTranformerInitialisation(
                    f"Bad model parameter: {name} with value {value} can't be parsed as a {type_}",
                    reason="MICROSERVICE_BAD_PARAMETER",
                )
            except KeyError:
                raise InvalidTranformerInitialisation(
                    f"Bad model parameter type: {type_}, valid are INT, FLOAT, DOUBLE, STRING, BOOL",
                    reason="MICROSERVICE_BAD_PARAMETER",
                )
    return parsed_parameters


def load_pipeline_from_settings(hf_settings: HuggingFaceSettings) -> Pipeline:
    """
    TODO
    """
    # TODO: Support URI for locally downloaded artifacts
    # uri = model_parameters.uri
    model = hf_settings.pretrained_model
    tokenizer = hf_settings.pretrained_tokenizer
    device = hf_settings.device

    if model and not tokenizer:
        tokenizer = model

    if hf_settings.optimum_model:
        optimum_class = SUPPORTED_OPTIMUM_TASKS[hf_settings.task]["class"][0]
        model = optimum_class.from_pretrained(
            hf_settings.pretrained_model,
            from_transformers=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # Device needs to be set to -1 due to known issue
        # https://github.com/huggingface/optimum/issues/191
        device = -1

    pp = pipeline(
        hf_settings.task,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=hf_settings.batch_size,
    )

    # If batch_size > 0 we need to ensure tokens are padded
    if hf_settings.batch_size:
        pp.tokenizer.pad_token_id = [str(pp.model.config.eos_token_id)]  # type: ignore

    return pp


def get_img_bytes(img: Image.Image, mime_type: Optional[str] = "png") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=mime_type)
    return buf.getvalue()


def b64encode_image(img: Image.Image, mime_type: Optional[str] = "png") -> str:
    return base64.b64encode(get_img_bytes(img, mime_type=mime_type)).decode()


def build_image(data: bytes) -> "Image.Image":
    return Image.open(io.BytesIO(data))


def serialize_conversation(obj: "Conversation") -> Dict:
    return {
        "conversation_id": str(obj.uuid),
        "past_user_inputs": obj.past_user_inputs,
        "generated_responses": obj.generated_responses,
        "new_user_input": obj.new_user_input,
    }


def deserialize_conversation(data: Dict) -> "Conversation":
    kwargs = {}
    if "conversation_id" in data:
        kwargs["conversation_id"] = uuid.UUID(data["conversation_id"])
    if "new_user_input" in data:
        kwargs["text"] = data["new_user_input"]
    if "past_user_inputs" in data:
        kwargs["past_user_inputs"] = data["past_user_inputs"]
    return Conversation(**kwargs)


class CommonJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(str(obj))
        elif isinstance(obj, (np.int_, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, Image.Image):
            return b64encode_image(obj)
        elif isinstance(obj, Conversation):
            return serialize_conversation(obj)
        else:
            return json.JSONEncoder.default(self, obj)


class MetaInfo:
    def __init__(self):
        self.metadata = {
            "audio-classification": dict(
                inputs=[
                    self._input_type("inputs", "base64", "BYTES"),
                ],
                outputs=[],
            ),
            "automatic-speech-recognition": dict(
                inputs=[
                    self._input_type("inputs", "base64", "BYTES"),
                ],
                outputs=[],
            ),
            "feature-extraction": dict(
                inputs=[
                    self._input_type("inputs", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "text-classification": dict(
                inputs=[
                    self._input_type("args", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "token-classification": dict(
                inputs=[
                    self._input_type("inputs", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "question-answering": dict(
                inputs=[
                    self._input_type("context", "str", "BYTES", is_single=True),
                    self._input_type("question", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "table-question-answering": dict(
                inputs=[
                    self._input_type(
                        "table", CONTENT_TYPE_JSON, "BYTES", is_single=True
                    ),
                    self._input_type("query", "str", "BYTES"),
                ],
                outputs=[],
            ),
            # only sinle supported now
            "visual-question-answering": dict(
                inputs=[
                    self._input_type(
                        "image", CONTENT_TYPE_IMAGE, "BYTES", is_single=True
                    ),
                    self._input_type("question", "str", "BYTES", is_single=True),
                ],
                outputs=[],
            ),
            "fill-mask": dict(
                inputs=[
                    self._input_type("inputs", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "summarization": dict(
                inputs=[
                    self._input_type("args", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "translation": dict(
                inputs=[
                    self._input_type("args", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "text2text-generation": dict(
                inputs=[
                    self._input_type("args", CONTENT_TYPE_JSON, "BYTES"),
                ],
                outputs=[],
            ),
            "text-generation": dict(
                inputs=[
                    self._input_type("args", CONTENT_TYPE_JSON, "BYTES"),
                ],
                outputs=[],
            ),
            "zero-shot-classification": dict(
                inputs=[
                    self._input_type("sequences", "str", "BYTES"),
                    self._input_type("candidate_labels", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "zero-shot-image-classification": dict(
                inputs=[
                    self._input_type("images", CONTENT_TYPE_IMAGE, "BYTES"),
                    self._input_type("candidate_labels", "str", "BYTES"),
                ],
                outputs=[],
            ),
            "conversational": dict(
                inputs=[
                    self._input_type(
                        "conversations", CONTENT_TYPE_CONVERSATION, "BYTES"
                    ),
                ],
                outputs=[],
            ),
            "image-classification": dict(
                inputs=[
                    self._input_type("images", CONTENT_TYPE_IMAGE, "BYTES"),
                ],
                outputs=[],
            ),
            "image-segmentation": dict(
                inputs=[
                    self._input_type("inputs", CONTENT_TYPE_IMAGE, "BYTES"),
                ],
                outputs=[],
            ),
            "object-detection": dict(
                inputs=[
                    self._input_type("images", CONTENT_TYPE_IMAGE, "BYTES"),
                ],
                outputs=[],
            ),
        }

    def _input_type(
        self, name, content_type, datatype: str, is_single: bool = False
    ) -> Dict:
        return dict(
            name=name,
            parameters=dict(
                content_type=content_type,
                is_single=is_single,
            ),
            datatype=datatype,
            shape=[],
        )

    def metainfo_for(self, task: str) -> Dict:
        return self.metadata.get(task, {})


metainfo_instance = MetaInfo()

metainfo_for = metainfo_instance.metainfo_for
