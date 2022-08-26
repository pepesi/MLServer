import json
from typing import Optional, Type, Any, Dict, List, Union
from functools import partial
from xmlrpc.client import Boolean
from mlserver.codecs.utils import (
    InputOrOutput,
    has_decoded,
    _save_decoded,
    is_list_of,
    get_decoded_or_raw,
    is_single_value,
    set_content_type,
)
from mlserver.codecs.base import (
    RequestCodec,
    register_request_codec,
    register_input_codec,
    find_input_codec,
    find_input_codec_by_payload,
    InputCodec as InputCodecTy,
)
from mlserver.codecs import (
    StringCodec,
)
from mlserver.codecs.base64 import _decode_base64, _encode_base64
from mlserver.codecs.pack import unpack
from mlserver.types import RequestInput, ResponseOutput
from PIL import Image

from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
)
from transformers.pipelines import Conversation

from mlserver.types.dataplane import Parameters
from .common import (
    CommonJSONEncoder,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_IMAGE,
    CONTENT_TYPE_CONVERSATION,
    deserialize_conversation,
    get_img_bytes,
    build_image,
    metainfo_for,
)


def get_input_codec(item: InputOrOutput, default: Optional[Type[InputCodecTy]]):
    if item.parameters is None:
        return default
    if not item.parameters.content_type:
        return default
    return find_input_codec(item.parameters.content_type)


def find_input_codec_by_payload_content(data: Any):
    """
    for huggingface JSONCodec has high priority
    """
    if JSONCodec.can_encode(data):
        return JSONCodec
    if ImageCodec.can_encode(data):
        return ImageCodec
    if ConversationCodec.can_encode(data):
        return ConversationCodec
    return find_input_codec_by_payload(data)


@register_input_codec
class ImageCodec(InputCodecTy):
    ContentType = CONTENT_TYPE_IMAGE

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return is_list_of(payload, Image.Image) or isinstance(payload, Image.Image)

    @classmethod
    def encode_output(
        cls,
        name: str,
        payload: List[Image.Image],
        use_bytes: Boolean = True,
        is_single=False,
        **kwargs,
    ) -> ResponseOutput:
        data = list(map(get_img_bytes, payload))
        packed = list(map(partial(_encode_base64, use_bytes=use_bytes), data))
        shape = [len(data)]
        output = ResponseOutput(
            name=name,
            datatype="BYTES",
            parameters=Parameters(content_type=cls.ContentType, is_single=is_single),
            shape=shape,
            data=packed,
        )
        return output

    @classmethod
    def decode_output(
        cls, response_output: ResponseOutput
    ) -> Union[List[Image.Image], Image.Image]:
        packed = response_output.data.__root__
        if is_single_value(response_output):
            if len(packed) == 0:
                raise ValueError("can't decode empty data as a single value")
            return build_image(next(map(_decode_base64, unpack(packed))))
        return list(map(build_image, map(_decode_base64, unpack(packed))))

    @classmethod
    def encode_input(
        cls,
        name: str,
        payload: List[Image.Image],
        use_bytes: Boolean = True,
        is_single: Boolean = True,
        **kwargs,
    ) -> RequestInput:
        output = cls.encode_output(
            name, payload, use_bytes=use_bytes, is_single=is_single, **kwargs
        )
        return RequestInput(
            name=output.name,
            parameters=output.parameters,
            shape=output.shape,
            data=output.data,
            datatype=output.datatype,
        )

    @classmethod
    def decode_input(
        cls, request_input: RequestInput
    ) -> Union[List[Image.Image], Image.Image]:
        packed = request_input.data.__root__
        if is_single_value(request_input):
            if len(packed) == 0:
                raise ValueError("can't decode empty data as a single value")
            return build_image(next(map(_decode_base64, unpack(packed))))
        return list(map(build_image, map(_decode_base64, unpack(packed))))


@register_input_codec
class JSONCodec(StringCodec):

    ContentType = CONTENT_TYPE_JSON

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            or is_list_of(payload, dict)
            or is_list_of(payload, list)
        )

    @classmethod
    def encode_output(cls, name: str, payload: Any, **kwargs) -> ResponseOutput:
        if type(payload) in (dict, Conversation):
            payload = [payload]
        json_strs = [json.dumps(el, cls=CommonJSONEncoder) for el in payload]
        output = StringCodec.encode_output(name, json_strs, **kwargs)
        set_content_type(output, cls.ContentType)
        return output

    @classmethod
    def decode_output(cls, response_output: ResponseOutput) -> Any:
        strs = StringCodec.decode_output(response_output)
        if type(strs) is list:
            return [json.loads(el) for el in strs]
        return json.loads(strs)

    @classmethod
    def encode_input(
        cls, name: str, payload: List[Dict[str, Any]], is_single=False, **kwargs
    ) -> RequestInput:
        if type(payload) is dict:
            is_single = True
            payload = [payload]
        strs = [json.dumps(el, cls=CommonJSONEncoder) for el in payload]
        _input = StringCodec.encode_input(name, strs, is_single=is_single, **kwargs)
        set_content_type(_input, cls.ContentType)
        return _input

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> Any:
        strs = StringCodec.decode_input(request_input)
        if type(strs) is list:
            return [json.loads(el) for el in strs]
        return json.loads(strs)


@register_input_codec
class ConversationCodec(StringCodec):
    ContentType = CONTENT_TYPE_CONVERSATION

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return isinstance(payload, Conversation) or is_list_of(payload, Conversation)

    @classmethod
    def encode_output(cls, name: str, payload: Any, **kwargs) -> ResponseOutput:
        if isinstance(payload, Conversation):
            payload = [payload]
        strs = [json.dumps(el, cls=CommonJSONEncoder) for el in payload]
        output = StringCodec.encode_output(name, strs)
        set_content_type(output, cls.ContentType)
        return output

    @classmethod
    def decode_output(cls, response_output: ResponseOutput) -> Any:
        strs = StringCodec.decode_input(response_output)
        if type(strs) is list:
            return [deserialize_conversation(json.loads(line)) for line in strs]
        return deserialize_conversation(json.loads(strs))

    @classmethod
    def encode_input(
        cls, name: str, payload: List[Conversation], **kwargs
    ) -> RequestInput:
        is_single = False
        if isinstance(payload, Conversation):
            is_single = True
            payload = [payload]
        strs = [json.dumps(el, cls=CommonJSONEncoder) for el in payload]
        _input = StringCodec.encode_input(name, strs, is_single=is_single, **kwargs)
        set_content_type(_input, cls.ContentType)
        return _input

    @classmethod
    def decode_input(cls, request_input: RequestInput) -> List[Conversation]:
        strs = StringCodec.decode_input(request_input)
        if type(strs) is list:
            return [deserialize_conversation(json.loads(line)) for line in strs]
        return deserialize_conversation(json.loads(strs))


@register_request_codec
class HuggingfaceRequestCodec(RequestCodec):

    InputCodec: Optional[Type[InputCodecTy]] = JSONCodec

    @classmethod
    def can_encode(cls, payload: Any) -> bool:
        return type(payload) in (dict, list, Conversation)

    @classmethod
    def encode_response(
        cls,
        model_name: str,
        payload: Union[List, Dict],
        model_version: Optional[str] = None,
        **kwargs,
    ) -> InferenceResponse:
        codec = find_input_codec_by_payload_content(payload)
        if codec is None:
            raise ValueError(f"can't encode {payload}, no codec found")
        is_single_value = type(payload) is not list
        if is_single_value:
            payload = [payload]
        outputs = [
            codec.encode_output(
                "output", payload=payload, is_single=is_single_value, **kwargs
            )
        ]
        return InferenceResponse(
            model_name=model_name,
            model_version=model_version,
            outputs=outputs,
        )

    @classmethod
    def decode_response(cls, response: InferenceResponse) -> Any:
        values = {}
        for item in response.outputs:
            if not has_decoded(item):
                codec = get_input_codec(item, cls.InputCodec)
                if not codec:
                    raise ValueError(f"can't decode {item}, no codec found")
                decoded_payload = codec.decode_output(item)
                _save_decoded(item, decoded_payload)
            value = get_decoded_or_raw(item)
            values[item.name] = value
        if "output" in values and len(values.keys()) == 1:
            return values["output"]
        return values

    @classmethod
    def _parameters_for(cls, task, input_args):
        meta = metainfo_for(task)
        if not meta:
            return {}
        for _input in meta.get("inputs", []):
            if _input["name"] == input_args:
                return _input.get("parameters", {})
        return {}

    @classmethod
    def encode_request(
        cls,
        payload: Dict[str, Any],
        is_single: Boolean = False,
        task: str = None,
        **kwargs,
    ) -> InferenceRequest:
        datas = []
        for key, value in payload.items():
            parameters = cls._parameters_for(task, key)
            if parameters.get("content_type"):
                codec = find_input_codec(parameters.get("content_type"))
            else:
                codec = find_input_codec_by_payload_content(value)
            if codec is None:
                raise ValueError(f"can't encode {key}: value is {value}")
            datas.append(codec.encode_input(key, value, is_single=is_single, **kwargs))
        return InferenceRequest(
            inputs=datas,
        )

    @classmethod
    def decode_request(cls, request: InferenceRequest) -> Dict[str, Any]:
        values = {}
        for item in request.inputs:
            codec = get_input_codec(item, cls.InputCodec)
            if not has_decoded(item) and codec is not None:
                decoded_payload = codec.decode_input(item)
                _save_decoded(item, decoded_payload)

            value = get_decoded_or_raw(item)
            values[item.name] = value
        return values
