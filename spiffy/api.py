import asyncio
from typing import List, Optional, Dict, Union, AsyncGenerator
from enum import Enum
import aiohttp
import json
import uuid
import websockets
import spiffy


def _get_headers():
    if spiffy.via_blobr:
        return {'X-BLOBR-KEY': spiffy.api_key}
    else:
        return {'Authorization': f'Bearer {spiffy.api_key}'}


async def _api_requester(api_url, method, data=None):
    """
    Makes an API request to the Spiffy API.
    args:
        api_url: The URL to make the request to.
        method: The HTTP method to use, either 'post' or 'get'.
        data: The data to send with the request. Should be None for 'get' requests.
    returns:
        The response text.
    """

    if method not in ['post', 'get']:
        raise ValueError("method must be either 'post' or 'get'")
    if method == 'get' and data is not None:
        raise ValueError("data must be None for 'get' requests")

    headers = _get_headers()

    async with aiohttp.ClientSession(headers=headers) as session:
        if method == 'get':
            async with session.get(api_url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    response_text = await response.text()
                    raise RuntimeError(f"Request failed with status code: {response.status} - {response_text}")
        elif method == 'post':
            async with session.post(api_url, json=data) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    response_text = await response.text()
                    raise RuntimeError(f"Request failed with status code: {response.status} - {response_text}")
        else:
            raise RuntimeError("This should never happen")


class TrainingStatus(Enum):
    """
    Potential statuses of a training job.
    """
    QUEUED = "QUEUED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


async def acreate_model(model_id: Optional[str] = None) -> str:
    """
    Creates a new user. If model_id is None, a random ID will be generated.
    If model_id already exists, the existing user_id will be returned.
    args:
        model_id: The ID of the user to create. If None, a random ID will be generated.
    returns:
        The ID of the created/model user.
    throws:
        RuntimeError: For all errors.
    """

    url = f"{spiffy.api_base_train}/users"
    response = await _api_requester(url, method='post', data={"userId": model_id})
    response = json.loads(response)
    if 'user' not in response:
        raise RuntimeError(f"'model' is missing from response: {response}")
    if 'id' not in response['user']:
        raise RuntimeError(f"'model.id' is missing from response: {response}")
    return response['user']['id']


async def _aget_inference_token(model: str) -> str:
    """
    Creates a new token used to authenticate with the inference service.
    args:
        model_id: The ID of the model to authenticate for.
    returns:
        The idToken of the model user.
    throws:
        RuntimeError: For all errors.
    """

    url = f"{spiffy.api_base_train}/auth/token"
    response = await _api_requester(url, method='post', data={"user_id": model})
    response = json.loads(response)
    if 'token' not in response:
        raise RuntimeError(f"'token' is missing from response: {response}")
    return response['token']


def _split_into_chunks(data, max_size):
    current_chunk = []
    current_chunk_size = 0

    for item in data:
        item_size = len(item.encode())  # Get the size of the string in bytes

        if current_chunk_size + item_size <= max_size:
            current_chunk.append(item)
            current_chunk_size += item_size
        else:
            yield current_chunk
            current_chunk = [item]
            current_chunk_size = item_size

    if current_chunk:
        yield current_chunk


async def _upload_chunk(base_url, chunk):
    response = await _api_requester(f"{base_url}/upload", method='post', data={'data': chunk})
    response = json.loads(response)
    if "sliceId" not in response:
        raise RuntimeError(f"'sliceId' is missing from response: {response}")
    else:
        return response['sliceId']


async def aupload_training_data(model_id: str, training_data: List[str]) -> str:
    """
    Uploads training data for a model.
    args:
        model_id: The ID of the user to upload training data for.
        training_data: The training data to upload.
    returns:
        data_id: The ID of the uploaded training data.
    throws:
        ValueError: If the model ID doesn't exist.
        RuntimeError: For any other error.
    """
    max_chunk_size = 900 * 1024  # 900KB in bytes
    base_url = f"{spiffy.api_base_train}/users/{model_id}/blobs"

    uploads = []
    chunks = []
    for i, chunk in enumerate(_split_into_chunks(training_data, max_chunk_size)):
        training_data = [x.replace('\n', '<new_line>') for x in chunk]
        upload = _upload_chunk(base_url, training_data)
        uploads.append(upload)

    responses = await asyncio.gather(*uploads)
    for i, chunk in enumerate(responses):
        chunks.append(chunk)

    response = await _api_requester(f"{base_url}/merge", method='post', data={'slices': chunks})
    response = json.loads(response)
    if "data_id" not in response:
        raise RuntimeError(f"'data_id' is missing from response: {response}")
    return response['data_id']


async def atrain(model_id: str, train_config: Dict[str, Union[str, int, float]] = None) -> str:
    """
    Trains a model.
    args:
        model_id: The ID of the model to train.
        train_config: The training configuration.
    returns:
        train_id: The ID of the training job. This can be used to check the status of the training job.
    throws:
        ValueError: If the model doesn't exist.
        RuntimeError: For any other error.
    """
    default_train_config = {
        "machineType": "a2-highgpu-1g",  # one gpu
        'checkpoint_every_n_epochs': 100000,  # to disable intermediate checkpoints
        'train_dev_split': 0.0,  # to train on the whole training data
    }
    if train_config is None:
        train_config = {}
    for key, value in default_train_config.items():
        if key not in train_config:
            train_config[key] = value

    url = f"{spiffy.api_base_train}/tasks/train"
    response = await _api_requester(url, method='post', data={
        'userId': model_id,
        'runConfig': json.dumps({'train_config': train_config}),
    })
    response = json.loads(response)
    if 'trainId' not in response:
        raise RuntimeError(f"'trainId' is missing from response: {response}")
    return response['trainId']


async def aget_train_status(train_id: str) -> TrainingStatus:
    """
    Gets status of a training job.
    args:
        train_id: The ID of the training job.
    returns:
        status: The status of the training job.
    throws:
        ValueError: If train_id doesn't exist.
        RuntimeError: For any other error.
    """

    url = f"{spiffy.api_base_train}/tasks/status"
    response = await _api_requester(url, method='post', data={'trainId': train_id})
    response = json.loads(response)
    if 'status' not in response:
        raise RuntimeError("Response doesn't contain 'status' key")
    status = TrainingStatus(response['status'])
    return status


async def _websocket_caller(
        model: str, model_v: str, prompt: str,
        generation_config: Dict[str, Union[str, int, float]],
        fallback_to_default_model: bool = False) -> AsyncGenerator[List[List[str]], None]:
    """
    The main inference function. Returns a streamed list of generated tokens
    args:
        model: The ID of the user.
        model_v: Model version as returned by `aget_available_user_models`
    returns:
        output (List[List[str]]): A list of generations, each list is a list of tokens, each token is a string.
            This is a streamed output. It can be used as follows:
            ```
            for output in agenerate(...):
                print(output)
            ```
    throws:
        ValueError: If model or model_v don't exist.
        RuntimeError: For any other error.
    """
    text_message = {
        "action": "decode",
        "input_text": prompt,
        "stopping_tokens": ["####", "</s>"],
        "max_decode_length": 20,
        "diversity_penalty": 0.99,
    }
    text_message.update(generation_config)
    uid = str(uuid.uuid1())
    text_message["id"] = uid

    websocket_url = spiffy.api_base_infer.replace("http", "ws", 1)
    websocket_url = f'{websocket_url}{model_v}'

    token = await _aget_inference_token(model)
    auth_message = {'action': 'auth', 'idToken': token}

    async with websockets.connect(websocket_url, open_timeout=300, ping_interval=300, ping_timeout=300) as client:
        for input_message in [auth_message, text_message]:
            await client.send(json.dumps(input_message))
            while True:
                output = await client.recv()
                output_decoded = json.loads(output)
                if 'id' in input_message:
                    message_id = output_decoded.get("id", None)
                    assert message_id == uid

                if output_decoded["response"]["type"] == "generation_complete":
                    break
                elif output_decoded["response"]["type"] == "status":
                    # loading model / finished loading messags
                    if output_decoded["response"]["message"].startswith('Finished loading'):
                        if model not in output_decoded["response"]["message"] and not fallback_to_default_model:
                            raise ModuleNotFoundError(f"User {model} doesn't have a model {model_v}")

                elif output_decoded["response"]["type"] == "error":
                    raise RuntimeError(f"Error connecting to the inference service: {output_decoded}")
                elif output_decoded["response"]["type"] == "success":
                    # successful authentication
                    break
                else:
                    if "tokens" not in output_decoded["response"]:
                        raise RuntimeError(
                            f"Response {output_decoded['response']} doesn't contain 'tokens': {output_decoded}")
                    yield output_decoded["response"]["tokens"]


async def aget_available_user_models(user_id: str) -> List[str]:
    """
    Gets a list of the available models for a user. Model versions will be sorted by training time
    args:
        user_id: The ID of the user.
    returns:
        List of available models for the user.
    throws:
        ValueError: If user_id doesn't exist.
        RuntimeError: For any other error.
    """
    url = f"{spiffy.api_base_infer}/model_available?spiffy_uid={user_id}"
    response = await _api_requester(url, method='get')
    model_v = response
    try:
        # The `model_available` endpoint returns the default model even if the user doesn't have a personalized model
        # so we need to check if the model exists by trying to generate a token from the websocket
        await _websocket_caller(user_id, model_v, prompt="test",
                                generation_config={"max_decode_length": 1},
                                fallback_to_default_model=False).__anext__()

    except ModuleNotFoundError:
        return []

    return [response]


async def agenerate(model_id: str, model_v: str, prompt: str,
                    generation_config: Dict[str, Union[str, int, float]]) -> AsyncGenerator[List[List[str]], None]:
    """
    The main inference function. Returns a streamed list of generated tokens
    args:
        model: The ID of the user.
        model_v: Model version as returned by `aget_available_user_models`
    returns:
        output (List[List[str]]): A list of generations, each list is a list of tokens, each token is a string.
            This is a streamed output. It can be used as follows:
            ```
            for output in agenerate(...):
                print(output)
            ```
    throws:
        ValueError: If user_id or model_v don't exist.
        RuntimeError: For any other error.
    """
    async for x in _websocket_caller(model_id, model_v, prompt, generation_config, fallback_to_default_model=False):
        yield x
