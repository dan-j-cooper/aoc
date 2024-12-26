from asyncio.futures import Future
from codecs import StreamWriter
from dataclasses import dataclass
from enum import StrEnum
import re
import asyncio
from inspect import Signature
from typer.params import Option
import uvloop
from ollama import AsyncClient, chat
from ollama import ChatResponse
import json
import functools
import orjson
from typing import Annotated, Callable, Any, TypedDict, Sequence, Literal
import asyncio
import pytest
import uvloop
import typer

from llama import system_prompt

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


def options() -> dict[str, Any]:
    return {
        "temperature": 0.05,
        "top_p": 0.3,
        "num_predict": 512,
        "num_ctx": int(32e3),
    }


class LlamaClient:
    def __init__(self, model: str, url: str = "/chat/completions"):
        self.host = "127.0.0.1"
        self.port = 5050
        base_url = f"http://{self.host}:{self.port}/v1"
        self.headers = {
            "Context-Type": "text/json",
            "Authorization": f"Bearer Bearer no-key",
        }
        self.url = base_url + url
        self.template = {
            "model": model,
            "cache_prompt": True,
            "messages": [],
            "params": options(),
        }
        self.rx: asyncio.StreamReader | None = None
        self.tx: asyncio.StreamWriter | None = None

    async def connect(self):
        self.rx, self.tx = await asyncio.open_connection(self.host, self.port)

    async def chat(self, msg: dict) -> Future[]:
        await self.tx.write(orjson.dumps(msg))
        return await orjson.loads(self.rx.read())


class Refactor(TypedDict):
    input: str
    target: str
    context: str


def system_prompt() -> Message:
    primer = """You are a precise code refactoring assistant that transforms Python function calls.
    Your responses will be in JSON format with careful attention to:
    1. Exact parameter name preservation
    2. Proper string escaping in JSON
    3. Using only variables available in the provided context
    4. Maintaining the exact function signature from the target template
    5. Use knowledge of python semantics to simplify code if possible given only CONTEXT 
    6. Respond with complete and valid python syntax"""

    # First, provide a clear example with explicit parameter mapping
    example1 = Refactor(
        input="""def get_lowpass_sos(freq: float, order: int, fs: float, zero_phase=False) -> NDArray:
                 return signal.butter(N=order, Wn=freq, btype="low", analog=False, output="sos", fs=fs)""",
        target="""scipy.signal.butter(N, Wn, btype="low", analog=False, output="sos", fs=None)""",
        context="""get_lowpass_sos(f, o, s)""",
    )

    example2 = Refactor(
        input='''def row_vect(arr: NDArray | list | tuple):
                    """convert a 1D array or list to a 1xN vector"""
                    assert_nd(arr, 1, "arr")
                    return np.array(arr).reshape(1, len(arr))''',
        target="np.array(x).reshape(1, -1)",
        context="""zoomies = np.zeros(10)
                    row_vect(zoomies)""",
    )
    example1_response = '{ "update": "scipy.signal.butter(N=o, Wn=f, btype="low", analog=False, output="sos", fs=s)" }'
    example2_response = '{ "update": "zoomies.reshape(1,-1)" }'

    example1_explanation = """
    Parameter Mapping in Example:
    - 'freq' (input) → 'f' (context) → 'Wn' (target)
    - 'order' (input) → 'o' (context) → 'N' (target)
    - 'fs' (input) → 's' (context) → 'fs' (target)
    """

    example2_explanation = """
    Parameter Mapping in Example:
    - 'arr' (input) → 'zoomies' (context) → 'x' (target)
    - since in CONTEXT zoomies is a numpy array, we can simplify be removing np.array(zoomies) and instead call zoomies.reshape(1, -1)
    """

    task_description = """
    TASK REQUIREMENTS:
    1. Analyze the INPUT function to understand its parameter structure
    2. Map the variables from CONTEXT to the corresponding TARGET parameters
    3. Generate a valid call to TARGET using only variables from CONTEXT
    4. Preserve exact parameter names and order from TARGET
    5. Return response in format: { "update": "transformed_code" }
    6. If transformation is impossible, return: { "update": { "error": "detailed_explanation" } }
    
    VALIDATION RULES:
    - All variables must exist in CONTEXT
    - Parameter names must exactly match TARGET
    - String values must be properly escaped for JSON
    - No new variables or literals may be introduced
    """

    content = f"""
    {primer}

    {task_description}
    
    EXAMPLE INPUT: {json.dumps(example1)}

    {example1_explanation}
    EXAMPLE OUTPUT: {example1_response}

    EXAMPLE INPUT: {json.dumps(example2)}
    {example2_explanation}
    EXAMPLE OUTPUT: {example2_response}
    """
    return Message(role="system", content=re.sub(r" +", " ", content))


def user_prompt(task: Refactor) -> Message:
    prompt = f"CURRENT TASK: {json.dumps(task)}"
    return Message(role="user", content=prompt)


async def refactor(task: Refactor, client: AsyncClient):
    system = system_prompt()
    user = user_prompt(task)
    messages = [system, user]
    resp = await client.chat(
        model="llama3.3:70b-instruct-q4_K_M",
        messages=messages,
        options=options(),
        format="json",
        cache_prompt=True,
    )
    return orjson.loads(resp.message.content)["update"]


@pytest.fixture
def test_client():
    client = AsyncClient()
    return client


@pytest.mark.asyncio
async def test_lowpass_sos(test_client):
    input = """def get_lowpass_sos(freq: float, order: int, fs: float, zero_phase=False) -> NDArray:
                #Compute IIR butterworth lowpass sos filter coefficients, for use by sosfiltfilt()
                #if zero_phase is True (order must be multiple of 2), or for use by sosfilt()
                #if zero_phase is False.
                if zero_phase:
                    order = _validate_zero_phase_order_single_side(order)
                return signal.butter(N=order, Wn=freq, btype="low", analog=False, output="sos", fs=fs)"""
    target = """scipy.signal.butter(N, Wn, btype="low", analog=False, output="sos", fs=None)"""
    context = "get_lowpass_sos(a, b, c)"
    request = Refactor(input=input, target=target, context=context)
    want = (
        'scipy.signal.butter(N=b, Wn=a, btype="low", analog=False, output="sos", fs=c)'
    )
    got = await refactor(request, test_client)

    assert want == got, f"{want}\n{got}"


@pytest.mark.asyncio
async def test_naive(test_client):
    input = '''def naive_r_nd(i_data, q_data):
    """
    Handles Naive R calculation using Euclidean distance of I and Q
    for separated I and Q arrays with any dimensions.
    """
    # return norm(np.stack((i_data, q_data), axis=0), axis=0) # SLOOWWWWWW
    # about twice as fast for 1d size 100-1000
    return np.sqrt(i_data**2 + q_data**2)'''
    target = "np.sqrt(i_data**2 + q_data**2)"
    context = "naive_r_nd(zam, zoom)"
    request = Refactor(input=input, target=target, context=context)
    want = "np.sqrt(zam**2 + zoom**2)"
    got = await refactor(request, test_client)

    assert want == got, f"{want}\n{got}"


async def server():
    print("hi")


def main(
    generate_system_prompt: Annotated[bool, typer.Option("-s", "--sys")] = False,
):
    if generate_system_prompt:
        prompt = system_prompt()
        print(orjson.dumps(prompt))
    else:
        asyncio.run(server())


if __name__ == "__main__":
    typer.run(main)
